import { useState, useEffect } from "react";
import Box from "@mui/material/Box";
import Container from "@mui/material/Container";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import Alert from "@mui/material/Alert";
import Fade from "@mui/material/Fade";
import ToggleButton from "@mui/material/ToggleButton";
import ToggleButtonGroup from "@mui/material/ToggleButtonGroup";
import Tooltip from "@mui/material/Tooltip";
import AutoAwesomeIcon from "@mui/icons-material/AutoAwesome";
import StorageIcon from "@mui/icons-material/Storage";
import CloudDownloadIcon from "@mui/icons-material/CloudDownload";

import Header from "./components/Header";
import UploadZone from "./components/UploadZone";
import LevelSelector from "./components/LevelSelector";
import SummaryView from "./components/SummaryView";
import LoadingOverlay from "./components/LoadingOverlay";

import { summarizePdf, type SummaryResponse } from "./services/api";

type AppState = "idle" | "loading" | "done";
type ModelSource = "finetuned" | "huggingface";

export default function App() {
  const [state, setState] = useState<AppState>("idle");
  const [file, setFile] = useState<File | null>(null);
  const [level, setLevel] = useState(1);
  const [modelSource, setModelSource] = useState<ModelSource>("finetuned");
  const [result, setResult] = useState<SummaryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Redirect any non-root path back to / (no router installed)
  useEffect(() => {
    if (window.location.pathname !== "/") {
      window.history.replaceState(null, "", "/");
    }
  }, []);

  const handleSubmit = async () => {
    if (!file) return;
    setError(null);
    setState("loading");

    try {
      const data = await summarizePdf(file, level, level === 4 ? "finetuned" : modelSource);
      setResult(data);
      setState("done");
    } catch (err) {
      setError(err instanceof Error ? err.message : "An unexpected error occurred.");
      setState("idle");
    }
  };

  const handleReset = () => {
    setFile(null);
    setResult(null);
    setError(null);
    setModelSource("finetuned");
    setState("idle");
  };

  return (
    <Box sx={{ minHeight: "100vh", bgcolor: "background.default" }}>
      <Header />

      <Container maxWidth="md" sx={{ py: { xs: 4, md: 6 } }}>
        {state === "done" && result ? (
          <SummaryView data={result} onReset={handleReset} />
        ) : (
          <Fade in timeout={400}>
            <Box>
              {/* Hero text */}
              <Box sx={{ textAlign: "center", mb: 5 }}>
                <Typography
                  variant="h4"
                  sx={{
                    mb: 1.5,
                    background: "linear-gradient(135deg, #1a237e 0%, #534bae 100%)",
                    WebkitBackgroundClip: "text",
                    WebkitTextFillColor: "transparent",
                  }}
                >
                  Summarize Legal PDFs
                </Typography>
                <Typography variant="subtitle1" sx={{ maxWidth: 520, mx: "auto" }}>
                  Upload a case document, choose your summary depth, and get an
                  AI-generated analysis in seconds.
                </Typography>
              </Box>

              {/* Error alert */}
              {error && (
                <Alert
                  severity="error"
                  onClose={() => setError(null)}
                  sx={{ mb: 3, borderRadius: 2 }}
                >
                  {error}
                </Alert>
              )}

              {/* Upload zone */}
              <Box sx={{ mb: 4 }}>
                <UploadZone file={file} onFileSelect={setFile} />
              </Box>

              {/* Level selector */}
              <Box sx={{ mb: 4 }}>
                <LevelSelector value={level} onChange={setLevel} />
              </Box>

              {/* Model source toggle (hidden for Extractive) */}
              {level !== 4 && (
                <Box sx={{ mb: 4 }}>
                  <Typography
                    variant="subtitle2"
                    sx={{ mb: 1.5, fontWeight: 600, color: "text.primary" }}
                  >
                    Model Source
                  </Typography>
                  <ToggleButtonGroup
                    value={modelSource}
                    exclusive
                    onChange={(_, v) => v && setModelSource(v as ModelSource)}
                    size="small"
                    fullWidth
                    sx={{ borderRadius: 2.5 }}
                  >
                    <ToggleButton
                      value="finetuned"
                      sx={{
                        borderRadius: "12px 0 0 12px",
                        textTransform: "none",
                        fontWeight: 600,
                        gap: 1,
                      }}
                    >
                      <Tooltip title="Uses locally saved fine-tuned model" arrow>
                        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                          <StorageIcon fontSize="small" />
                          Finetuned Model
                        </Box>
                      </Tooltip>
                    </ToggleButton>
                    <ToggleButton
                      value="huggingface"
                      sx={{
                        borderRadius: "0 12px 12px 0",
                        textTransform: "none",
                        fontWeight: 600,
                        gap: 1,
                      }}
                    >
                      <Tooltip title="Downloads base model from HuggingFace Hub" arrow>
                        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                          <CloudDownloadIcon fontSize="small" />
                          HuggingFace Model
                        </Box>
                      </Tooltip>
                    </ToggleButton>
                  </ToggleButtonGroup>
                </Box>
              )}

              {/* Submit button */}
              <Box sx={{ textAlign: "center" }}>
                <Button
                  variant="contained"
                  size="large"
                  onClick={handleSubmit}
                  disabled={!file}
                  startIcon={<AutoAwesomeIcon />}
                  sx={{
                    px: 5,
                    py: 1.5,
                    fontSize: "1rem",
                    borderRadius: 3,
                  }}
                >
                  Generate Summary
                </Button>
              </Box>
            </Box>
          </Fade>
        )}
      </Container>

      <LoadingOverlay open={state === "loading"} />
    </Box>
  );
}
