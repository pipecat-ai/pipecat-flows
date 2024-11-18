export function setupCanvas(graph) {
  const canvas = new LGraphCanvas('#graph-canvas', graph);

  function resizeCanvas() {
    const canvasElement = document.getElementById('graph-canvas');
    const container = document.getElementById('graph-container');
    canvasElement.width = container.offsetWidth;
    canvasElement.height = container.offsetHeight;
  }

  window.addEventListener('resize', resizeCanvas);
  resizeCanvas();

  return canvas;
}
