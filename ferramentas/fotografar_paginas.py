"""Fotografa páginas do app com o Chromium sem janela, para a conferência visual.

Por que não `chrome --headless --screenshot --virtual-time-budget=...`: com esse comando o
Chromium fotografa o esqueleto do Streamlit antes de o websocket entregar a página, e todo
gráfico sai vazio. Este script fala com o Chromium pelo DevTools (CDP), espera o Streamlit
terminar de rodar e cada gráfico Vega desenhar, estica a janela até a altura do conteúdo e
só então fotografa a página inteira.

Uso, com o app rodando (por exemplo `streamlit run app.py --server.port 8599`):

    python ferramentas/fotografar_paginas.py <url> <saida.png> [pasta_recortes]

    python ferramentas/fotografar_paginas.py "http://localhost:8599/populacao?ver=estado&aba=condicoes" p1.png

- `?ver=regiao|estado` escolhe o nível dos Painéis 1 e 2; `?aba=<atalho>` abre uma aba
  (onde, condicoes, qualidade; h1 a h5, comparacao, variabilidade, limitacoes).
- Com `pasta_recortes`, grava também recortes de 1440 x 1600 px (precisa do Pillow), mais
  fáceis de ler que a página inteira.
- O Chromium vem da variável de ambiente `CHROME`; sem ela, procura o do Playwright em
  `%LOCALAPPDATA%\\ms-playwright`. Precisa do pacote `websockets`.
- Só fotografa depois que o aviso do rodapé (`TEXTO_FINAL`, o último elemento de toda página)
  aparece: o Streamlit entrega os elementos aos poucos, e o rodapé pode chegar depois de os
  gráficos já estarem desenhados.
- Imprime um resumo em JSON (gráficos visíveis e desenhados, exceções na página, altura, se o
  aviso final chegou) e os avisos do console.
"""
import asyncio
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import websockets

PORTA_CDP = 9333
LARGURA = 1440
# O começo do aviso que fecha toda página (tela.AVISO_FONTES): a página só está completa com ele.
TEXTO_FINAL = "Trabalho acadêmico independente"

ESTADO_JS = """
(() => {
  const graficos = [...document.querySelectorAll('[data-testid="stVegaLiteChart"]')];
  const visiveis = graficos.filter(g => g.offsetParent !== null);
  const desenhados = visiveis.filter(g => g.querySelector('canvas, svg.marks')).length;
  const rodando = !!document.querySelector('[data-testid="stStatusWidget"]');
  const esqueleto = document.querySelectorAll('[data-testid="stSkeleton"]').length;
  const principal = document.querySelector('[data-testid="stMainBlockContainer"]');
  const altura = principal ? principal.getBoundingClientRect().bottom + 120 : 0;
  const erros = [...document.querySelectorAll('[data-testid="stException"]')].length;
  const fim = (principal ? principal.innerText : '').includes(TEXTO_FINAL);
  return JSON.stringify({graficos: visiveis.length, desenhados, rodando, esqueleto, altura, erros, fim,
                         titulo: (document.querySelector('h1') || {}).innerText || ''});
})()
"""


def achar_chrome() -> str:
    """O executável do Chromium: a variável `CHROME` ou o mais novo do Playwright."""
    if os.environ.get("CHROME"):
        return os.environ["CHROME"]
    base = Path(os.environ.get("LOCALAPPDATA", "")) / "ms-playwright"
    candidatos = sorted(base.glob("chromium-*/chrome-win64/chrome.exe"))
    if not candidatos:
        sys.exit("Chromium não encontrado: defina a variável de ambiente CHROME.")
    return str(candidatos[-1])


async def fotografar(url: str, saida: Path, recortes: Path | None) -> None:
    perfil = Path(tempfile.mkdtemp(prefix="foto-cdp-"))
    chrome = subprocess.Popen([
        achar_chrome(), "--headless=new", "--disable-gpu", "--hide-scrollbars",
        f"--remote-debugging-port={PORTA_CDP}", f"--user-data-dir={perfil}",
        f"--window-size={LARGURA},1200", "about:blank",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        # Espera o Chromium abrir a aba: logo depois de outra foto, a porta pode responder
        # ainda sem nenhuma página.
        ws_url = None
        for _ in range(150):
            try:
                alvos = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORTA_CDP}/json").read())
                ws_url = next((a["webSocketDebuggerUrl"] for a in alvos if a["type"] == "page"), None)
            except OSError:
                pass
            if ws_url:
                break
            time.sleep(0.2)
        if not ws_url:
            raise RuntimeError(f"O Chromium não abriu uma página na porta {PORTA_CDP}.")
        async with websockets.connect(ws_url, max_size=200_000_000) as ws:
            contador = 0
            console: list[str] = []

            async def cdp(metodo: str, **params):
                nonlocal contador
                contador += 1
                await ws.send(json.dumps({"id": contador, "method": metodo, "params": params}))
                while True:
                    msg = json.loads(await ws.recv())
                    if msg.get("id") == contador:
                        return msg.get("result", {})
                    if msg.get("method") == "Runtime.consoleAPICalled" and msg["params"]["type"] in ("warning", "error"):
                        texto = " ".join(str(a.get("value", a.get("description", ""))) for a in msg["params"]["args"])
                        console.append(texto[:300])
                    elif msg.get("method") == "Runtime.exceptionThrown":
                        console.append("EXC " + json.dumps(msg["params"]["exceptionDetails"])[:300])

            async def estado() -> dict:
                expressao = ESTADO_JS.replace("TEXTO_FINAL", json.dumps(TEXTO_FINAL))
                r = await cdp("Runtime.evaluate", expression=expressao, returnByValue=True)
                return json.loads(r["result"]["value"])

            async def esperar(limite: float = 90) -> dict:
                """Espera a página parar: Streamlit sem rodar, sem esqueleto, todo gráfico desenhado,
                o aviso final na tela, e o mesmo estado em quatro leituras seguidas."""
                estavel, ultimo, inicio = 0, None, time.time()
                while time.time() - inicio < limite:
                    await asyncio.sleep(0.5)
                    e = await estado()
                    pronto = (not e["rodando"] and e["esqueleto"] == 0 and e["titulo"] and e["fim"]
                              and e["desenhados"] == e["graficos"])
                    chave = (e["graficos"], e["desenhados"], round(e["altura"]))
                    estavel = estavel + 1 if (pronto and chave == ultimo) else 0
                    ultimo = chave
                    if estavel >= 4:
                        return e
                return await estado()

            async def janela(altura: int) -> None:
                await cdp("Emulation.setDeviceMetricsOverride", width=LARGURA, height=altura,
                          deviceScaleFactor=1, mobile=False)

            await janela(1200)
            await cdp("Page.enable")
            await cdp("Runtime.enable")
            await cdp("Page.navigate", url=url)
            e = await esperar()
            for limite in (30,):  # a página muda de altura ao esticar a janela: espera de novo
                await janela(int(min(max(e["altura"], 1200), 30000)))
                e = await esperar(limite)
            altura = int(min(max(e["altura"], 1200), 30000))
            await janela(altura)
            await asyncio.sleep(1.5)
            foto = await cdp("Page.captureScreenshot", format="png", captureBeyondViewport=False)
            saida.write_bytes(base64.b64decode(foto["data"]))
            print(json.dumps({**e, "altura_final": altura}, ensure_ascii=False))
            for linha in dict.fromkeys(console):
                print("console:", linha)
    finally:
        chrome.terminate()
        try:
            chrome.wait(timeout=10)
        except subprocess.TimeoutExpired:
            chrome.kill()
        # O Chromium encerra os processos filhos com atraso; o perfil temporário que sobrar
        # fica para a limpeza normal da pasta Temp.
        shutil.rmtree(perfil, ignore_errors=True)
    if recortes:
        from PIL import Image

        recortes.mkdir(parents=True, exist_ok=True)
        imagem = Image.open(saida)
        for i, topo in enumerate(range(0, imagem.height, 1600), start=1):
            imagem.crop((0, topo, LARGURA, min(topo + 1600, imagem.height))).save(recortes / f"{saida.stem}-{i}.png")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    asyncio.run(fotografar(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]) if len(sys.argv) > 3 else None))
