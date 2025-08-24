// preprocess.js (ESM)
console.log("✅ preprocess v2 loaded");
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ---- Load bundle ----
const BUNDLE_PATH = path.join(__dirname, "preprocessing_bundle.json");
if (!fs.existsSync(BUNDLE_PATH)) {
  throw new Error("preprocessing_bundle.json not found next to server.js");
}
const BUNDLE = JSON.parse(fs.readFileSync(BUNDLE_PATH, "utf-8"));

export const SELECTED_FEATURES = BUNDLE.selected_features || [];
const LABEL_MAPS = BUNDLE.label_encoders || {};          // { col: { mapping, unknown_index } }
const SCALER = BUNDLE.scaler || { mean_: null, scale_: null }; // arrays aligned with SELECTED_FEATURES

// ---- Helpers ----
const trim = (s) => String(s ?? "").trim();
const isFiniteNum = (x) => typeof x === "number" && Number.isFinite(x);
const toNum = (v) => (isFiniteNum(v) ? v : (Number.isFinite(Number(v)) ? Number(v) : 0));
const onlyICD = (s) => (trim(s).match(/[A-Z][0-9][0-9A-Z.]*/i)?.[0] || "Unknown").toUpperCase();
const onlyCPT = (s) => (trim(s).match(/\b\d{5}\b/)?.[0] || "00000");

// Encode categorical value per bundle mapping with unknown fallback
function encodeCategorical(colName, rawValue) {
  const entry = LABEL_MAPS[colName];
  if (!entry) return null; // not a label-encoded col
  const mapping = entry.mapping || {};
  const unknownIdx = entry.unknown_index ?? null;

  const key = (rawValue === null || rawValue === undefined || rawValue === "")
    ? "Unknown"
    : String(rawValue);

  if (Object.prototype.hasOwnProperty.call(mapping, key)) return mapping[key];
  if (unknownIdx !== null && unknownIdx !== undefined) return unknownIdx;
  if (Object.prototype.hasOwnProperty.call(mapping, "__UNK__")) return mapping["__UNK__"];
  return 0;
}

// ---- Normalize UI payload -> training feature names ----
function normalizeToTraining(raw) {
  // Accept both your UI keys (provider_id, diagnosis_code, etc.) and direct training keys.
  const provider = raw.Provider ?? trim(raw.provider_id);
  const diag = raw.ClmDiagnosisCode_1 ?? onlyICD(raw.diagnosis_code);
  const proc = raw.ClmProcedureCode_1 ?? onlyCPT(raw.procedure_code);
  const loc = raw.DiagnosisGroupCode ?? trim(raw.service_location);

  return {
    Provider: provider || "Unknown",
    InscClaimAmtReimbursed: toNum(raw.InscClaimAmtReimbursed ?? String(raw.claim_amount ?? 0).replace(/[^\d.]/g, "")),
    AttendingPhysician: raw.AttendingPhysician ?? (trim(raw.referring_physician) || "Unknown"),
    OperatingPhysician: raw.OperatingPhysician ?? "Unknown",
    OtherPhysician: raw.OtherPhysician ?? "Unknown",
    ClmAdmitDiagnosisCode: raw.ClmAdmitDiagnosisCode ?? diag,
    DeductibleAmtPaid: toNum(raw.DeductibleAmtPaid ?? raw.deductible_amount ?? 0),
    DiagnosisGroupCode: loc || "Unknown",

    ClmDiagnosisCode_1: diag,
    ClmDiagnosisCode_2: raw.ClmDiagnosisCode_2 ?? "Unknown",
    ClmDiagnosisCode_3: raw.ClmDiagnosisCode_3 ?? "Unknown",
    ClmDiagnosisCode_4: raw.ClmDiagnosisCode_4 ?? "Unknown",
    ClmDiagnosisCode_5: raw.ClmDiagnosisCode_5 ?? "Unknown",
    ClmDiagnosisCode_6: raw.ClmDiagnosisCode_6 ?? "Unknown",
    ClmDiagnosisCode_7: raw.ClmDiagnosisCode_7 ?? "Unknown",
    ClmDiagnosisCode_8: raw.ClmDiagnosisCode_8 ?? "Unknown",
    ClmDiagnosisCode_9: raw.ClmDiagnosisCode_9 ?? "Unknown",
    ClmDiagnosisCode_10: raw.ClmDiagnosisCode_10 ?? "Unknown",

    ClmProcedureCode_1: proc,
    ClmProcedureCode_2: raw.ClmProcedureCode_2 ?? "Unknown",
    ClmProcedureCode_3: raw.ClmProcedureCode_3 ?? "Unknown",
    ClmProcedureCode_4: raw.ClmProcedureCode_4 ?? "Unknown",
    ClmProcedureCode_5: raw.ClmProcedureCode_5 ?? "Unknown",

    Gender: raw.Gender ?? (raw.gender ?? "Unknown"),
    Race: raw.Race ?? (raw.race ?? "Unknown"),
    RenalDiseaseIndicator: raw.RenalDiseaseIndicator ?? (raw.renal ?? "Unknown"),
    State: raw.State ?? (raw.state ?? "Unknown"),
    County: raw.County ?? (raw.county ?? "Unknown"),

    NoOfMonths_PartACov: toNum(raw.NoOfMonths_PartACov ?? 12),
    NoOfMonths_PartBCov: toNum(raw.NoOfMonths_PartBCov ?? 12),

    ChronicCond_Alzheimer: toNum(raw.ChronicCond_Alzheimer ?? 0),
    ChronicCond_Heartfailure: toNum(raw.ChronicCond_Heartfailure ?? 0),
    ChronicCond_KidneyDisease: toNum(raw.ChronicCond_KidneyDisease ?? 0),
    ChronicCond_Cancer: toNum(raw.ChronicCond_Cancer ?? 0),
    ChronicCond_ObstrPulmonary: toNum(raw.ChronicCond_ObstrPulmonary ?? 0),
    ChronicCond_Depression: toNum(raw.ChronicCond_Depression ?? 0),
    ChronicCond_Diabetes: toNum(raw.ChronicCond_Diabetes ?? 0),
    ChronicCond_IschemicHeart: toNum(raw.ChronicCond_IschemicHeart ?? 0),
    ChronicCond_Osteoporasis: toNum(raw.ChronicCond_Osteoporasis ?? 0),
    ChronicCond_rheumatoidarthritis: toNum(raw.ChronicCond_rheumatoidarthritis ?? 0),
    ChronicCond_stroke: toNum(raw.ChronicCond_stroke ?? 0),

    IPAnnualReimbursementAmt: toNum(raw.IPAnnualReimbursementAmt ?? 0),
    IPAnnualDeductibleAmt: toNum(raw.IPAnnualDeductibleAmt ?? 0),
    OPAnnualReimbursementAmt: toNum(raw.OPAnnualReimbursementAmt ?? 0),
    OPAnnualDeductibleAmt: toNum(raw.OPAnnualDeductibleAmt ?? 0),

    AgeAtClaim: toNum(raw.AgeAtClaim ?? raw.patient_age ?? 0),
    ClaimDuration: toNum(raw.ClaimDuration ?? Math.max(1, Number(raw.service_frequency ?? 1))),
    ClaimsPerProvider: toNum(raw.ClaimsPerProvider ?? 1),
    ClaimsPerBene: toNum(raw.ClaimsPerBene ?? 1),
  };
}

// ---- Core transform (encode + scale) ----
export function transformOne(raw) {
  const normalized = normalizeToTraining(raw);
  const out = new Array(SELECTED_FEATURES.length);

  // Fill with encoded/scaled values in exact order
  for (let i = 0; i < SELECTED_FEATURES.length; i++) {
    const col = SELECTED_FEATURES[i];
    if (LABEL_MAPS[col]) {
      out[i] = encodeCategorical(col, normalized[col]);
    } else {
      out[i] = toNum(normalized[col]);
    }
  }

  // Apply scaler by index (arrays are aligned to SELECTED_FEATURES)
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
  if (!Array.isArray(rows)) throw new Error("transformBatch expects an array of objects");
  return rows.map(transformOne);
}
