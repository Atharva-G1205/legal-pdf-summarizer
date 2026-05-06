import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import Paper from "@mui/material/Paper";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import InsertDriveFileIcon from "@mui/icons-material/InsertDriveFile";
import { alpha, useTheme } from "@mui/material/styles";

interface UploadZoneProps {
  file: File | null;
  onFileSelect: (file: File) => void;
}

export default function UploadZone({ file, onFileSelect }: UploadZoneProps) {
  const theme = useTheme();

  const onDrop = useCallback(
    (accepted: File[]) => {
      if (accepted.length > 0) onFileSelect(accepted[0]);
    },
    [onFileSelect],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "application/pdf": [".pdf"] },
    multiple: false,
    maxSize: 50 * 1024 * 1024, // 50 MB
  });

  const borderColor = isDragActive
    ? theme.palette.primary.main
    : file
      ? theme.palette.success.main
      : theme.palette.divider;

  const bgColor = isDragActive
    ? alpha(theme.palette.primary.main, 0.04)
    : file
      ? alpha(theme.palette.success.main, 0.03)
      : "transparent";

  return (
    <Paper
      {...getRootProps()}
      elevation={0}
      sx={{
        p: { xs: 4, md: 6 },
        border: "2px dashed",
        borderColor,
        borderRadius: 3,
        bgcolor: bgColor,
        cursor: "pointer",
        transition: "all 0.25s ease",
        "&:hover": {
          borderColor: "primary.light",
          bgcolor: alpha(theme.palette.primary.main, 0.03),
        },
        textAlign: "center",
      }}
    >
      <input {...getInputProps()} />

      {file ? (
        <Box sx={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 1.5 }}>
          <InsertDriveFileIcon sx={{ fontSize: 48, color: "success.main" }} />
          <Typography variant="h6" sx={{ fontSize: "1rem", color: "text.primary" }}>
            {file.name}
          </Typography>
          <Typography variant="body2" sx={{ color: "text.secondary" }}>
            {(file.size / 1024 / 1024).toFixed(2)} MB — Click or drag to replace
          </Typography>
        </Box>
      ) : (
        <Box sx={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 1.5 }}>
          <CloudUploadIcon
            sx={{
              fontSize: 52,
              color: isDragActive ? "primary.main" : "text.secondary",
              transition: "color 0.2s",
            }}
          />
          <Typography variant="h6" sx={{ fontSize: "1rem", color: "text.primary" }}>
            {isDragActive ? "Drop your PDF here" : "Drag & drop a legal PDF"}
          </Typography>
          <Typography variant="body2" sx={{ color: "text.secondary" }}>
            or click to browse — .pdf up to 50 MB
          </Typography>
        </Box>
      )}
    </Paper>
  );
}
