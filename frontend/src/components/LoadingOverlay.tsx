import Backdrop from "@mui/material/Backdrop";
import Box from "@mui/material/Box";
import CircularProgress from "@mui/material/CircularProgress";
import Typography from "@mui/material/Typography";

interface LoadingOverlayProps {
  open: boolean;
}

export default function LoadingOverlay({ open }: LoadingOverlayProps) {
  return (
    <Backdrop
      open={open}
      sx={{
        zIndex: (t) => t.zIndex.drawer + 1,
        bgcolor: "rgba(255,255,255,0.82)",
        backdropFilter: "blur(6px)",
        flexDirection: "column",
        gap: 3,
      }}
    >
      <CircularProgress size={52} thickness={4} sx={{ color: "primary.main" }} />

      <Box sx={{ textAlign: "center" }}>
        <Typography variant="h6" sx={{ fontSize: "1rem", mb: 0.5 }}>
          Analyzing document…
        </Typography>
        <Typography variant="body2" sx={{ color: "text.secondary", maxWidth: 320 }}>
          Running extractive ranking and generating the summary.
          This may take 30-60 seconds on first run.
        </Typography>
      </Box>
    </Backdrop>
  );
}
