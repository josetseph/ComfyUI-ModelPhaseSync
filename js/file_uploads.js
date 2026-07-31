/**
 * Upload buttons for Load Conditioning / Load Latent (Upload).
 * Load Image gets this natively via image_upload; .conditioning / .latent
 * need a custom widget that POSTs to /upload/image (raw bytes).
 */
const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;

function findWidget(node, name) {
  return node.widgets?.find((w) => w.name === name);
}

function ensureComboValue(widget, value, emptyPrefix) {
  if (!widget) return;
  const values = widget.options?.values;
  if (Array.isArray(values) && !values.includes(value)) {
    widget.options.values = values.filter(
      (v) => typeof v === "string" && !(emptyPrefix && v.startsWith(emptyPrefix))
    );
    widget.options.values.push(value);
  }
  widget.value = value;
  widget.callback?.(value);
}

async function uploadToInputSubfolder(file, subfolder) {
  const body = new FormData();
  body.append("image", file); // Comfy upload route always expects this field name
  body.append("overwrite", "true");
  body.append("type", "input");
  body.append("subfolder", subfolder);

  const resp = await api.fetchApi("/upload/image", { method: "POST", body });
  if (resp.status !== 200) {
    throw new Error(`Upload failed: ${resp.status} ${resp.statusText}`);
  }
  const data = await resp.json();
  // Return path relative to input/ (dropdown / annotated path style).
  return data.subfolder ? `${data.subfolder}/${data.name}` : data.name;
}

function addUploadButton(node, { accept, subfolder, comboName, emptyPrefix, label }) {
  const flag = `_mpsUpload_${comboName}`;
  if (node[flag]) return;
  node[flag] = true;

  const fileInput = document.createElement("input");
  fileInput.type = "file";
  fileInput.accept = accept;
  fileInput.style.display = "none";
  document.body.appendChild(fileInput);

  fileInput.addEventListener("change", async () => {
    const file = fileInput.files?.[0];
    fileInput.value = "";
    if (!file) return;
    try {
      const relPath = await uploadToInputSubfolder(file, subfolder);
      const combo = findWidget(node, comboName);
      const pathWidget = findWidget(node, "path");
      // Combo lists are often basename-only when rooted at the subfolder;
      // keep both: prefer basename for conditionings-style roots, full rel for latents.
      const comboValue =
        comboName === "conditioning_file" ? relPath.split("/").pop() : relPath;
      ensureComboValue(combo, comboValue, emptyPrefix);
      if (pathWidget) {
        pathWidget.value = relPath;
      }
      node.setDirtyCanvas?.(true, true);
    } catch (err) {
      console.error("[ModelPhaseSync] upload failed", err);
      alert(String(err?.message || err));
    }
  });

  node.addWidget(
    "button",
    label || "choose file to upload",
    null,
    () => fileInput.click(),
    { serialize: false }
  );

  const onRemoved = node.onRemoved;
  node.onRemoved = function () {
    fileInput.remove();
    return onRemoved?.apply(this, arguments);
  };
}

const UPLOAD_NODES = {
  LoadConditioning: {
    accept: ".conditioning,application/octet-stream",
    subfolder: "conditionings",
    comboName: "conditioning_file",
    emptyPrefix: "no .conditioning",
  },
  LoadLatentUpload: {
    accept: ".latent,application/octet-stream",
    subfolder: "latents",
    comboName: "latent_file",
    emptyPrefix: "no .latent",
  },
};

app.registerExtension({
  name: "ModelPhaseSync.FileUploads",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    const cfg = UPLOAD_NODES[nodeData.name];
    if (!cfg) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      addUploadButton(this, cfg);
      return r;
    };
  },
});
