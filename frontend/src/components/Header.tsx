import AppBar from "@mui/material/AppBar";
import Toolbar from "@mui/material/Toolbar";
import Typography from "@mui/material/Typography";
import GavelIcon from "@mui/icons-material/Gavel";
import Box from "@mui/material/Box";

export default function Header() {
  return (
    <AppBar
      position="sticky"
      color="inherit"
      sx={{
        bgcolor: "rgba(255,255,255,0.85)",
        backdropFilter: "blur(12px)",
        borderBottom: "1px solid",
        borderColor: "divider",
      }}
    >
      <Toolbar sx={{ gap: 1.5, px: { xs: 2, md: 4 } }}>
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            width: 38,
            height: 38,
            borderRadius: 2,
            bgcolor: "primary.main",
            color: "white",
          }}
        >
          <GavelIcon fontSize="small" />
        </Box>

        <Box>
          <Typography
            variant="h6"
            sx={{ fontSize: "1.1rem", lineHeight: 1.2, color: "text.primary" }}
          >
            Legal Summarizer
          </Typography>
          <Typography
            variant="caption"
            sx={{ color: "text.secondary", letterSpacing: "0.02em" }}
          >
            AI-powered PDF analysis
          </Typography>
        </Box>
      </Toolbar>
    </AppBar>
  );
}
