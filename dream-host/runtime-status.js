export function resolveRuntimeStatus({
  readyWorkerCount = 0,
  notebookComputeOnline = false,
  notebookChatWarmupEnabled = false,
  notebookChatModelReady = false,
  activeWorkerDevice = null,
  notebookChatModelDevice = null
} = {}) {
  const nativeWorkerReady = Number.isInteger(readyWorkerCount) && readyWorkerCount > 0;
  const notebookRuntimeAvailable = Boolean(notebookChatWarmupEnabled && notebookComputeOnline);
  const notebookModelReady = Boolean(notebookRuntimeAvailable && notebookChatModelReady);
  const deviceName = value => typeof value === "string" && value.trim() ? value.trim() : null;

  return {
    modelRuntime: nativeWorkerReady ? "gpu-worker" : (notebookRuntimeAvailable ? "dream-notebook" : "offline"),
    modelReady: nativeWorkerReady || notebookModelReady,
    chatAvailable: nativeWorkerReady || notebookRuntimeAvailable,
    device: nativeWorkerReady
      ? deviceName(activeWorkerDevice)
      : (notebookModelReady ? deviceName(notebookChatModelDevice) : null)
  };
}
