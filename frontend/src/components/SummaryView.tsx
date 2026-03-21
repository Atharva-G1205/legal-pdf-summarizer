import Box from "@mui/material/Box";
import Paper from "@mui/material/Paper";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import Chip from "@mui/material/Chip";
import Divider from "@mui/material/Divider";
import DownloadIcon from "@mui/icons-material/Download";
import RestartAltIcon from "@mui/icons-material/RestartAlt";
import Fade from "@mui/material/Fade";

import { type SummaryResponse, downloadSummaryPdf } from "../services/api";

interface SummaryViewProps {
  data: SummaryResponse;
  onReset: () => void;
}

export default function SummaryView({ data, onReset }: SummaryViewProps) {
  const handleDownload = async () => {
    await downloadSummaryPdf(data.summary, data.level_name, data.filename);
  };

  return (
    <Fade in timeout={500}>
      <Box sx={{ maxWidth: 780, mx: "auto", width: "100%" }}>
        {/* Header */}
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            flexWrap: "wrap",
            gap: 2,
            mb: 3,
          }}
        >
          <Box>
            <Typography variant="h5" sx={{ mb: 0.5 }}>
              {data.level_emoji} {data.level_name} Summary
            </Typography>
            <Typography variant="body2" sx={{ color: "text.secondary" }}>
              Source: <strong>{data.filename}</strong>
            </Typography>
          </Box>

          <Chip
            label={`${data.word_count} words`}
            size="small"
            sx={{
              fontWeight: 600,
              bgcolor: "primary.main",
              color: "white",
              px: 1,
            }}
          />
        </Box>

        {/* Summary content */}
        <Paper
          elevation={0}
          sx={{
            p: { xs: 3, md: 4 },
            borderRadius: 3,
            border: "1px solid",
            borderColor: "divider",
            mb: 3,
          }}
        >
          {data.summary.split("\n").map((line, i) => {
            const trimmed = line.trim();
            if (!trimmed) return <Box key={i} sx={{ height: 12 }} />;

            // Extractive numbered lines get special styling
            const isNumbered = /^\d+\.\s*\[/.test(trimmed);

            return (
              <Typography
                key={i}
                variant="body1"
                sx={{
                  mb: 1,
                  fontWeight: isNumbered ? 500 : 400,
                  color: isNumbered ? "text.primary" : "text.primary",
                  pl: isNumbered ? 1 : 0,
                  borderLeft: isNumbered ? "3px solid" : "none",
                  borderColor: "primary.light",
                  lineHeight: 1.75,
                }}
              >
                {trimmed}
              </Typography>
            );
          })}
        </Paper>

        <Divider sx={{ mb: 3 }} />

        {/* Actions */}
        <Box
          sx={{
            display: "flex",
            gap: 2,
            justifyContent: "center",
            flexWrap: "wrap",
          }}
        >
          <Button
            variant="contained"
            startIcon={<DownloadIcon />}
            onClick={handleDownload}
            size="large"
          >
            Download as PDF
          </Button>

          <Button
            variant="outlined"
            startIcon={<RestartAltIcon />}
            onClick={onReset}
            size="large"
          >
            Upload Another
          </Button>
        </Box>
      </Box>
    </Fade>
  );
}
