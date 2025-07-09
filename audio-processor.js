class PCMResamplerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._origBuffer = [];
    this._origSampleRate = sampleRate;
    this._targetSampleRate = 16000;
    this._ratio = this._origSampleRate / this._targetSampleRate;
    this._bufferSize = 4096;
  }

  process(inputs) {
    const input = inputs[0][0];
    if (input && input.length > 0) {
      this._origBuffer.push(new Float32Array(input));
      const neededOrig = Math.ceil(this._bufferSize * this._ratio);
      let totalOrig = 0;
      for (const buf of this._origBuffer) totalOrig += buf.length;

      if (totalOrig >= neededOrig) {
        const flat = new Float32Array(totalOrig);
        let offset = 0;
        for (const buf of this._origBuffer) {
          flat.set(buf, offset);
          offset += buf.length;
        }
        const leftover = flat.subarray(neededOrig);
        this._origBuffer = [new Float32Array(leftover)];

        const out = new Float32Array(this._bufferSize);
        for (let i = 0; i < this._bufferSize; i++) {
          const idx = Math.floor(i * this._ratio);
          out[i] = flat[idx];
        }
        this.port.postMessage(out);
      }
    }
    return true;
  }
}
registerProcessor('pcm-resampler-processor', PCMResamplerProcessor);