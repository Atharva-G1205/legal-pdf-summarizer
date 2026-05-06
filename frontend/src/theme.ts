import { createTheme } from "@mui/material/styles";

/**
 * Custom MUI theme — Refined Utilitarian (Google M3 aesthetic)
 *
 * Deep indigo primary, warm amber accent, crisp neutrals.
 * Uses Google's Product Sans / Roboto pairing.
 */
const theme = createTheme({
  palette: {
    mode: "light",
    primary: {
      main: "#1a237e",     // deep indigo
      light: "#534bae",
      dark: "#000051",
      contrastText: "#ffffff",
    },
    secondary: {
      main: "#ff8f00",     // warm amber
      light: "#ffc046",
      dark: "#c56000",
      contrastText: "#000000",
    },
    background: {
      default: "#f5f5f7",
      paper: "#ffffff",
    },
    text: {
      primary: "#1d1d1f",
      secondary: "#6e6e73",
    },
    divider: "rgba(0,0,0,0.08)",
    success: {
      main: "#2e7d32",
    },
    error: {
      main: "#d32f2f",
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica Neue", Arial, sans-serif',
    h4: {
      fontWeight: 700,
      letterSpacing: "-0.02em",
    },
    h5: {
      fontWeight: 600,
      letterSpacing: "-0.01em",
    },
    h6: {
      fontWeight: 600,
    },
    subtitle1: {
      fontWeight: 500,
      color: "#6e6e73",
    },
    body1: {
      fontSize: "0.95rem",
      lineHeight: 1.7,
    },
    button: {
      textTransform: "none",
      fontWeight: 600,
    },
  },
  shape: {
    borderRadius: 12,
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 10,
          padding: "10px 24px",
          fontSize: "0.9rem",
        },
        containedPrimary: {
          boxShadow: "0 2px 8px rgba(26,35,126,0.25)",
          "&:hover": {
            boxShadow: "0 4px 16px rgba(26,35,126,0.35)",
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          boxShadow: "0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06)",
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          boxShadow: "0 1px 3px rgba(0,0,0,0.06)",
        },
      },
    },
  },
});

export default theme;
