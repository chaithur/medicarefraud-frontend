// preprocess.js (ESM)
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const BUNDLE_PATH = path.join(__dirname, "preprocessing_bundle.json");
if (!fs.existsSync(BUNDLE_PATH)) {
  throw new Error("preprocessing_bundle.json not found next to server.js");
}
const BUNDLE = JSON.parse(fs.readFileSync(BUNDLE_PATH, "utf-8"));

const LABEL_MAPS = BUNDLE.label_encoders || {};
export const SELECTED_FEATURES = BUNDLE.selected_features || [];
const SCALER = BUNDLE.scaler || { mean_: null, scale_: null };

const isNumber = (x) => typeof x === "number" && Number.isFinite(x);
const toNumberOrZero = (v) => (isNumber(v) ? v : (Number.isFinite(Number(v)) ? Number(v) : 0));

function encodeCategorical(colName, rawValue) {
  const entry = LABEL_MAPS[colName];
  if (!entry) return null; // not a label-encoded column
  const mapping = entry.mapping || {};
  const unknownIdx = entry.unknown_index ?? null;

  const key = (rawValue === null || rawValue === undefined || rawValue === "") ? "Unknown" : String(rawValue);
  if (Object.prototype.hasOwnProperty.call(mapping, key)) return mapping[key];
  return (unknownIdx !== null && unknownIdx !== undefined) ? unknownIdx : 0;
}

export function transformOne(raw) {
  const out = new Array(SELECTED_FEATURES.length);

  for (let i = 0; i < SELECTED_FEATURES.length; i++) {
    const col = SELECTED_FEATURES[i];
    if (LABEL_MAPS[col]) {
      out[i] = encodeCategorical(col, raw?.[col]);
    } else {
      out[i] = toNumberOrZero(raw?.[col]);
    }
  }

  // Safe scaling: skip slots with null/NaN/Infinity or scale 0
  if (
    Array.isArray(SCALER.mean_) &&
    Array.isArray(SCALER.scale_) &&
    SCALER.mean_.length === out.length &&
    SCALER.scale_.length === out.length
  ) {
    for (let i = 0; i < out.length; i++) {
      const m = SCALER.mean_[i];
      const s = SCALER.scale_[i];
      if (!Number.isFinite(m) || !Number.isFinite(s) || s === 0) continue;
      out[i] = (out[i] - m) / s;
    }
  }

  return out;
}

export function transformBatch(rows) {
  if (!Array.isArray(rows)) {
    throw new Error("transformBatch expects an array of objects");
  }
  return rows.map(transformOne);
}
