// transformer.js
const fs = require("fs");
const path = require("path");

const BUNDLE = JSON.parse(fs.readFileSync(path.join(__dirname, "preprocessing_bundle.json"), "utf8"));
const FEATURES = BUNDLE.selected_features;         // 49 names, strict order
const ENCODERS = BUNDLE.label_encoders || {};      // { col: { mapping, unknown_index } }
const SCALER   = BUNDLE.scaler || {};              // { mean_: [...], scale_: [...] } aligned to FEATURES
const MEAN_    = SCALER.mean_ || [];
const SCALE_   = SCALER.scale_ || [];

const trim = (s) => String(s ?? "").trim();
const onlyICD = (s) => (trim(s).match(/[A-Z][0-9][0-9A-Z.]*/i)?.[0] || "Unknown").toUpperCase();
const onlyCPT = (s) => (trim(s).match(/\b\d{5}\b/)?.[0] || "00000");

// Encode categorical with fallback to unknown_index
function encode(col, val) {
  const enc = ENCODERS[col];
  if (!enc || !enc.mapping) return 0; // safe default
  const key = (val === null || val === undefined || val === "") ? "Unknown" : String(val);
  if (key in enc.mapping) return enc.mapping[key];
  if ("unknown_index" in enc) return enc.unknown_index;
  // last resort: if they used "__UNK__" as a key in mapping
  if ("__UNK__" in enc.mapping) return enc.mapping["__UNK__"];
  return 0;
}

// Scale numeric by index (because mean_/scale_ are arrays aligned to FEATURES)
function scaleByIndex(idx, val) {
  const m = MEAN_[idx], s = SCALE_[idx];
  if (m === null || s === null || s === 0 || s === undefined) {
    // not a numeric feature in the scaler → return raw number (or 0)
    return Number(val) || 0;
  }
  return (Number(val) - Number(m)) / Number(s);
}

// Map your UI/body to training feature names
function normalizeToTraining(raw) {
  return {
    Provider: trim(raw.provider_id),
    InscClaimAmtReimbursed: Number(String(raw.claim_amount || 0).replace(/[^\d.]/g, "")),
    AttendingPhysician: trim(raw.referring_physician) || "Unknown",
    OperatingPhysician: "Unknown",
    OtherPhysician: "Unknown",
    ClmAdmitDiagnosisCode: onlyICD(raw.diagnosis_code),
    DeductibleAmtPaid: Number(raw.deductible_amount) || 0,
    DiagnosisGroupCode: trim(raw.service_location) || "Unknown",

    ClmDiagnosisCode_1: onlyICD(raw.diagnosis_code),
    ClmDiagnosisCode_2: "Unknown",
    ClmDiagnosisCode_3: "Unknown",
    ClmDiagnosisCode_4: "Unknown",
    ClmDiagnosisCode_5: "Unknown",
    ClmDiagnosisCode_6: "Unknown",
    ClmDiagnosisCode_7: "Unknown",
    ClmDiagnosisCode_8: "Unknown",
    ClmDiagnosisCode_9: "Unknown",
    ClmDiagnosisCode_10: "Unknown",

    ClmProcedureCode_1: onlyCPT(raw.procedure_code),
    ClmProcedureCode_2: "Unknown",
    ClmProcedureCode_3: "Unknown",
    ClmProcedureCode_4: "Unknown",
    ClmProcedureCode_5: "Unknown",

    Gender: (raw.gender ?? "Unknown"),
    Race: (raw.race ?? "Unknown"),
    RenalDiseaseIndicator: (raw.renal ?? "Unknown"),
    State: (raw.state ?? "Unknown"),
    County: (raw.county ?? "Unknown"),

    NoOfMonths_PartACov: Number(raw.NoOfMonths_PartACov ?? 12),
    NoOfMonths_PartBCov: Number(raw.NoOfMonths_PartBCov ?? 12),

    ChronicCond_Alzheimer: Number(raw.ChronicCond_Alzheimer ?? 0),
    ChronicCond_Heartfailure: Number(raw.ChronicCond_Heartfailure ?? 0),
    ChronicCond_KidneyDisease: Number(raw.ChronicCond_KidneyDisease ?? 0),
    ChronicCond_Cancer: Number(raw.ChronicCond_Cancer ?? 0),
    ChronicCond_ObstrPulmonary: Number(raw.ChronicCond_ObstrPulmonary ?? 0),
    ChronicCond_Depression: Number(raw.ChronicCond_Depression ?? 0),
    ChronicCond_Diabetes: Number(raw.ChronicCond_Diabetes ?? 0),
    ChronicCond_IschemicHeart: Number(raw.ChronicCond_IschemicHeart ?? 0),
    ChronicCond_Osteoporasis: Number(raw.ChronicCond_Osteoporasis ?? 0),
    ChronicCond_rheumatoidarthritis: Number(raw.ChronicCond_rheumatoidarthritis ?? 0),
    ChronicCond_stroke: Number(raw.ChronicCond_stroke ?? 0),

    IPAnnualReimbursementAmt: Number(raw.IPAnnualReimbursementAmt ?? 0),
    IPAnnualDeductibleAmt: Number(raw.IPAnnualDeductibleAmt ?? 0),
    OPAnnualReimbursementAmt: Number(raw.OPAnnualReimbursementAmt ?? 0),
    OPAnnualDeductibleAmt: Number(raw.OPAnnualDeductibleAmt ?? 0),

    AgeAtClaim: Number(raw.patient_age ?? 0),
    ClaimDuration: Math.max(1, Number(raw.service_frequency ?? 1)),
    ClaimsPerProvider: Number(raw.ClaimsPerProvider ?? 1),
    ClaimsPerBene: Number(raw.ClaimsPerBene ?? 1),
  };
}

function buildVector(raw) {
  const named = normalizeToTraining(raw);

  // produce an encoded/scaled row in the exact order
  const row = FEATURES.map((col, idx) => {
    // if this column is present in label_encoders => treat as categorical
    if (ENCODERS[col]?.mapping) {
      return encode(col, named[col]);
    }
    // otherwise treat as numeric and scale by index
    return scaleByIndex(idx, named[col]);
  });

  // sanity guard: if lots of features explode, your map/scaler is mismatched
  const bad = row.filter(v => Number.isNaN(v) || Math.abs(v) > 50).length;
  if (bad) throw new Error("preprocessing_mismatch: check encoder mappings & scaler arrays.");
  return row;
}

module.exports = { FEATURES, buildVector };
