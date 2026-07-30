/**
 * Dynamic in_/out_ sockets for SyncBarrierN.
 * Starts with one pair; grows when the last input is connected; trims trailing
 * empty pairs down to a single buffer slot after the last connection.
 */
const { app } = window.comfyAPI.app;

const MAX_SLOTS = 1000;
const NODE_NAME = "SyncBarrierN";

function inputIndex(name) {
  const m = /^in_(\d+)$/.exec(name || "");
  return m ? parseInt(m[1], 10) : 0;
}

function ensurePairCount(node, count) {
  const target = Math.max(1, Math.min(MAX_SLOTS, count));

  while ((node.inputs?.length || 0) < target) {
    const i = (node.inputs?.length || 0) + 1;
    node.addInput(`in_${i}`, "*");
  }
  while ((node.outputs?.length || 0) < target) {
    const i = (node.outputs?.length || 0) + 1;
    node.addOutput(`out_${i}`, "*");
  }

  // Trim trailing unconnected inputs only (stop at first connected from end).
  while ((node.inputs?.length || 0) > target) {
    const last = node.inputs.length - 1;
    if (node.inputs[last]?.link != null) break;
    node.removeInput(last);
  }
  while ((node.outputs?.length || 0) > (node.inputs?.length || 0)) {
    const last = node.outputs.length - 1;
    if (node.outputs[last]?.links?.length) break;
    node.removeOutput(last);
  }
  while ((node.outputs?.length || 0) < (node.inputs?.length || 0)) {
    const i = node.outputs.length + 1;
    node.addOutput(`out_${i}`, "*");
  }
}

function desiredSlotCount(node) {
  let highestConnected = 0;
  for (let i = 0; i < (node.inputs?.length || 0); i++) {
    const inp = node.inputs[i];
    if (inp?.link == null) continue;
    const idx = inputIndex(inp.name) || i + 1;
    highestConnected = Math.max(highestConnected, idx);
  }
  // One buffer after the last connection; at least one slot when empty.
  return Math.max(1, Math.min(MAX_SLOTS, highestConnected + 1));
}

function stabilize(node) {
  if (node.mps_syncBarrierBusy) return;
  node.mps_syncBarrierBusy = true;
  try {
    ensurePairCount(node, desiredSlotCount(node));
  } finally {
    node.mps_syncBarrierBusy = false;
  }
}

app.registerExtension({
  name: "ModelPhaseSync.SyncBarrier",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    // Frontend starts with a single pair so new nodes are not born with 1000 pins.
    // Python still declares MAX_SLOTS optionals so dynamically added names validate.
    if (nodeData.input?.optional) {
      const first = nodeData.input.optional.in_1 || ["*"];
      nodeData.input.optional = { in_1: first };
    }
    if (Array.isArray(nodeData.output)) {
      nodeData.output = [nodeData.output[0] || "*"];
      nodeData.output_name = ["out_1"];
    }

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      if (!this.outputs?.length) {
        this.addOutput("out_1", "*");
      }
      if (!this.inputs?.length) {
        this.addInput("in_1", "*");
      }
      ensurePairCount(this, 1);
      return r;
    };

    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (type) {
      const r = onConnectionsChange?.apply(this, arguments);
      if (type === LiteGraph.INPUT) {
        stabilize(this);
      }
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (data) {
      const r = onConfigure?.apply(this, arguments);
      let maxIdx = 1;
      for (const inp of this.inputs || []) {
        maxIdx = Math.max(maxIdx, inputIndex(inp?.name) || 0);
      }
      if (data?.inputs) {
        for (const inp of data.inputs) {
          maxIdx = Math.max(maxIdx, inputIndex(inp?.name) || 0);
        }
      }
      ensurePairCount(this, Math.min(MAX_SLOTS, Math.max(1, maxIdx)));
      stabilize(this);
      return r;
    };
  },
});
