import { useState } from "react";
import Box from "@mui/material/Box";
import Container from "@mui/material/Container";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import Alert from "@mui/material/Alert";
import Fade from "@mui/material/Fade";
import AutoAwesomeIcon from "@mui/icons-material/AutoAwesome";

import Header from "./components/Header";
import UploadZone from "./components/UploadZone";
import LevelSelector from "./components/LevelSelector";
import SummaryView from "./components/SummaryView";
import LoadingOverlay from "./components/LoadingOverlay";

import { summarizePdf, type SummaryResponse } from "./services/api";

type AppState = "idle" | "loading" | "done";

export default function App() {
  const [state, setState] = useState<AppState>("idle");
  const [file, setFile] = useState<File | null>(null);
  const [level, setLevel] = useState(1);
  const [result, setResult] = useState<SummaryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async () => {
    if (!file) return;
    setError(null);
    setState("loading");

    try {
      const data = await summarizePdf(file, level);
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
