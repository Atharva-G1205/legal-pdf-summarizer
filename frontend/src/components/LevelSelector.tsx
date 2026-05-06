import { useEffect, useState } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Radio from "@mui/material/Radio";
import RadioGroup from "@mui/material/RadioGroup";
import FormControlLabel from "@mui/material/FormControlLabel";
import Paper from "@mui/material/Paper";
import Skeleton from "@mui/material/Skeleton";

import { fetchSummaryLevels, type SummaryLevel } from "../services/api";

interface LevelSelectorProps {
  value: number;
  onChange: (level: number) => void;
}

export default function LevelSelector({ value, onChange }: LevelSelectorProps) {
  const [levels, setLevels] = useState<SummaryLevel[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchSummaryLevels()
      .then(setLevels)
      .catch(() => {
        // Fallback if backend isn't running yet
        setLevels([
          { value: 1, name: "Executive", emoji: "📋", description: "Concise overview (~150 words)" },
          { value: 2, name: "Detailed", emoji: "📄", description: "Comprehensive summary (~400 words)" },
          { value: 3, name: "Technical", emoji: "🔬", description: "In-depth analysis (~600 words)" },
          { value: 4, name: "Extractive", emoji: "📌", description: "Top-ranked sentences, zero hallucination" },
        ]);
      })
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5 }}>
        {[1, 2, 3, 4].map((i) => (
          <Skeleton key={i} variant="rounded" height={56} sx={{ borderRadius: 2 }} />
        ))}
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="subtitle2" sx={{ mb: 1.5, fontWeight: 600, color: "text.primary" }}>
        Summary Level
      </Typography>

      <RadioGroup
        value={String(value)}
        onChange={(e) => onChange(Number(e.target.value))}
      >
        <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
          {levels.map((lvl) => (
            <Paper
              key={lvl.value}
              variant="outlined"
              sx={{
                px: 2,
                py: 1,
                borderRadius: 2.5,
                borderColor: value === lvl.value ? "primary.main" : "divider",
                borderWidth: value === lvl.value ? 2 : 1,
                bgcolor: value === lvl.value ? "primary.main" : "transparent",
                transition: "all 0.2s ease",
                cursor: "pointer",
                "&:hover": {
                  borderColor: "primary.light",
                },
              }}
              onClick={() => onChange(lvl.value)}
            >
              <FormControlLabel
                value={String(lvl.value)}
                control={
                  <Radio
                    size="small"
                    sx={{
                      color: value === lvl.value ? "white" : undefined,
                      "&.Mui-checked": { color: "white" },
                    }}
                  />
                }
                label={
                  <Box>
                    <Typography
                      variant="body2"
                      sx={{
                        fontWeight: 600,
                        color: value === lvl.value ? "white" : "text.primary",
                      }}
                    >
                      {lvl.emoji} {lvl.name}
                    </Typography>
                    <Typography
                      variant="caption"
                      sx={{
                        color: value === lvl.value
                          ? "rgba(255,255,255,0.8)"
                          : "text.secondary",
                      }}
                    >
                      {lvl.description}
                    </Typography>
                  </Box>
                }
                sx={{ m: 0, width: "100%" }}
              />
            </Paper>
          ))}
        </Box>
      </RadioGroup>
    </Box>
  );
}
