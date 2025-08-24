// server.js (ESM)
import cors from "cors";
import crypto from "crypto";
import "dotenv/config";
import express from "express";
import jwt from "jsonwebtoken";
import { MongoClient } from "mongodb";
import fetch from "node-fetch";

// Preprocessor: uses preprocessing_bundle.json next to this file
import { SELECTED_FEATURES, transformBatch } from "./preprocess.js";

const app = express();
app.use(cors({ origin: true, credentials: true }));
app.use(express.json({ limit: "4mb" }));

// ===== ENV =====
const {
  PORT = 8080,

  // Cosmos DB for MongoDB
  COSMOS_MONGO_URI,
  COSMOS_DB = "appdb",
  COSMOS_USERS = "users",

  // JWT
  JWT_SECRET,
  JWT_EXPIRES_IN = "1h",

  // Azure ML Claim Endpoint
  AML_CLAIM_URI,
  AML_CLAIM_KEY,
  AML_CLAIM_DEPLOYMENT,
  // choose: mlflow_split | mlflow | inputs
  AML_CLAIM_PAYLOAD_STYLE = "mlflow_split",

  // (Provider endpoint stubbed for now)
  AML_PROVIDER_URI,
  AML_PROVIDER_KEY,
  AML_PROVIDER_DEPLOYMENT
} = process.env;

// ===== Mongo (Cosmos for Mongo API) =====
let usersCollection = null;
if (!COSMOS_MONGO_URI) {
  console.warn("⚠️ COSMOS_MONGO_URI not set — /auth routes will fail until you add it to .env");
} else {
  const mongo = new MongoClient(COSMOS_MONGO_URI);
  mongo
    .connect()
    .then(() => {
      usersCollection = mongo.db(COSMOS_DB).collection(COSMOS_USERS);
      console.log("Mongo connected");
    })
    .catch((err) => {
      console.error("Mongo init error:", err);
      process.exit(1);
    });
}

// ===== Utils =====
const hashPassword = (pw) => crypto.createHash("sha256").update(String(pw)).digest("hex");

async function callAML(uri, key, deployment, payload) {
  if (!uri) throw new Error("AML scoring URI not set");
  const headers = { "Content-Type": "application/json", Accept: "application/json" };
  if (key) headers["Authorization"] = `Bearer ${key}`;
  if (deployment) headers["azureml-model-deployment"] = deployment;

  const r = await fetch(uri, { method: "POST", headers, body: JSON.stringify(payload) });
  const text = await r.text();
  if (!r.ok) throw new Error(`AML ${r.status}: ${text}`);
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

// ===== JWT =====
function signToken(payload) {
  if (!JWT_SECRET) throw new Error("JWT_SECRET not set");
  return jwt.sign(payload, JWT_SECRET, { expiresIn: JWT_EXPIRES_IN });
}
function auth(req, res, next) {
  const hdr = req.headers.authorization || "";
  const token = hdr.startsWith("Bearer ") ? hdr.slice(7) : null;
  if (!token) return res.status(401).json({ error: "Missing/invalid Authorization header" });
  try {
    req.user = jwt.verify(token, JWT_SECRET);
    next();
  } catch {
    res.status(401).json({ error: "Invalid or expired token" });
  }
}

// ===== Routes =====
app.get("/", (_req, res) =>
  res.send("Backend running 🚀 Try /health, /auth/signup, /auth/login, /predict/claim, /debug/transform-claim")
);
app.get("/health", (_req, res) => res.json({ ok: true }));

// --- Auth
app.post("/auth/signup", async (req, res) => {
  try {
    if (!usersCollection) throw new Error("Mongo not configured");
    const { email, password, name = "" } = req.body || {};
    if (!email || !password) return res.status(400).json({ error: "email & password required" });

    const id = String(email).toLowerCase();
    await usersCollection.insertOne({
      _id: id,
      email: id,
      name,
      pw: hashPassword(password),
      createdAt: new Date(),
    });

    const token = signToken({ email: id });
    res.json({ ok: true, token });
  } catch (e) {
    if (String(e?.code) === "11000") return res.status(409).json({ error: "email exists" });
    res.status(500).json({ error: e.message });
  }
});

app.post("/auth/login", async (req, res) => {
  try {
    if (!usersCollection) throw new Error("Mongo not configured");
    const { email, password } = req.body || {};
    const id = String(email || "").toLowerCase();
    const user = await usersCollection.findOne({ _id: id });
    if (!user || user.pw !== hashPassword(password)) {
      return res.status(401).json({ error: "invalid creds" });
    }
    const token = signToken({ email: id });
    res.json({ ok: true, token });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// --- Claim-level predict (accepts a single object or an array)
app.post("/predict/claim", auth, async (req, res) => {
  try {
    if (!req.body || typeof req.body !== "object") {
      return res.status(400).json({
        error: "bad-request",
        message: "Send JSON body: either a single object or an array of objects",
      });
    }
    const rawRows = Array.isArray(req.body) ? req.body : [req.body];

    // Normalize + encode + scale + order (done inside preprocess.js)
    const X = transformBatch(rawRows);

    // Build AML payload
    let payload;
    switch ((AML_CLAIM_PAYLOAD_STYLE || "mlflow_split").toLowerCase()) {
      case "mlflow":
        payload = { input_data: { columns: SELECTED_FEATURES, data: X } };
        break;
      case "inputs":
        payload = { inputs: X };
        break;
      default: // mlflow_split
        payload = {
          input_data: {
            columns: SELECTED_FEATURES,
            index: Array.from({ length: X.length }, (_, i) => i),
            data: X,
          },
        };
    }

    const out = await callAML(AML_CLAIM_URI, AML_CLAIM_KEY, AML_CLAIM_DEPLOYMENT, payload);
    res.json({
      ok: true,
      predictions: out,
      meta: { rows: X.length, cols: SELECTED_FEATURES.length, payloadStyle: AML_CLAIM_PAYLOAD_STYLE },
    });
  } catch (e) {
    res.status(502).json({
      error: "claim-scorer-failed",
      message: e.message,
      suggestions: [
        "Ensure preprocessing_bundle.json is beside server.js.",
        "If AML complains about schema, try AML_CLAIM_PAYLOAD_STYLE=mlflow_split (default), then mlflow, then inputs.",
      ],
    });
  }
});

// --- Debug: returns the numeric vectors we send to AML
app.post("/debug/transform-claim", auth, (req, res) => {
  try {
    const rawRows = Array.isArray(req.body) ? req.body : [req.body];
    const X = transformBatch(rawRows);
    res.json({
      ok: true,
      selected_features: SELECTED_FEATURES,
      vectors: X,
      rows: X.length,
      cols: SELECTED_FEATURES.length,
    });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// --- Provider-level predict (stub until aggregator is provided)
app.post("/predict/provider", auth, async (_req, res) => {
  res.json({ ok: true, note: "Provider route stubbed. Send the provider aggregation code to wire it." });
});

// server.js
import {
  SELECTED_FEATURES,
  transformBatch,
  diagnostics,
  PREPROCESS_VERSION
} from "./preprocess.js";

// …after app.get("/health", …) add:
app.get("/diag/preprocess", (_req, res) => {
  try {
    const d = diagnostics();
    res.json({ ok: true, ...d, selected_features_head: SELECTED_FEATURES.slice(0, 5) });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

app.listen(Number(PORT), () => console.log(`server.js listening on http://localhost:${PORT}`));
