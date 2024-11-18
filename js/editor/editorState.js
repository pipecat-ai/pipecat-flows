class EditorState {
  static instance = null;

  constructor() {
    if (EditorState.instance) {
      return EditorState.instance;
    }
    this.sidePanel = null;
    EditorState.instance = this;
  }

  setSidePanel(sidePanel) {
    this.sidePanel = sidePanel;
  }

  updateSidePanel(node) {
    if (this.sidePanel) {
      this.sidePanel.updatePanel(node);
    }
  }
}

export const editorState = new EditorState();
