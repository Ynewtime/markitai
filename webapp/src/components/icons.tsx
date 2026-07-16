import {
  ArrowCounterClockwise,
  ArrowRight,
  CaretRight,
  Check,
  ClockCounterClockwise,
  Desktop,
  DownloadSimple,
  Eye,
  EyeSlash,
  FilePdf,
  FileText,
  Gear,
  Globe,
  ArrowSquareOut,
  MagicWand,
  Moon,
  Sun,
  TerminalWindow,
  Trash,
  UploadSimple,
  Warning,
  X,
} from "@phosphor-icons/react";

/** One icon family throughout the app. Phosphor regular weight matches the
 * existing restrained 1.5px outline voice. The brand mark remains custom. */
interface IconProps {
  size?: number;
}

const iconProps = {
  weight: "regular" as const,
  "aria-hidden": true,
};

export function FileTextIcon({ size = 14 }: IconProps) {
  return <FileText size={size} {...iconProps} />;
}

export function GlobeIcon({ size = 14 }: IconProps) {
  return <Globe size={size} {...iconProps} />;
}

export function UploadIcon({ size = 14 }: IconProps) {
  return <UploadSimple size={size} {...iconProps} />;
}

export function DownloadIcon({ size = 15 }: IconProps) {
  return <DownloadSimple size={size} {...iconProps} />;
}

/** Brand tile: #18181b rounded square + white zigzag M (fixed in both themes). */
export function LogoMark({ size = 24, className }: IconProps & { className?: string }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-hidden="true"
    >
      <rect width="32" height="32" rx="8" fill="#18181b" />
      <rect
        x="0.5"
        y="0.5"
        width="31"
        height="31"
        rx="7.5"
        fill="none"
        strokeWidth="1"
        style={{ stroke: "var(--logo-edge)" }}
      />
      <path
        d="M8 23V9L13 17L16 11L19 17L24 9V23"
        stroke="white"
        strokeWidth="2.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />
    </svg>
  );
}

export function MonitorIcon({ size = 14 }: IconProps) {
  return <Desktop size={size} {...iconProps} />;
}

export function SunIcon({ size = 14 }: IconProps) {
  return <Sun size={size} {...iconProps} />;
}

export function MoonIcon({ size = 14 }: IconProps) {
  return <Moon size={size} {...iconProps} />;
}

export function HistoryIcon({ size = 16 }: IconProps) {
  return <ClockCounterClockwise size={size} {...iconProps} />;
}

export function XIcon({ size = 16 }: IconProps) {
  return <X size={size} {...iconProps} />;
}

export function TerminalIcon({ size = 15 }: IconProps) {
  return <TerminalWindow size={size} {...iconProps} />;
}

export function RotateCcwIcon({ size = 13 }: IconProps) {
  return <ArrowCounterClockwise size={size} {...iconProps} />;
}

export function MagicWandIcon({ size = 14 }: IconProps) {
  return <MagicWand size={size} {...iconProps} />;
}

export function ArrowRightIcon({ size = 13 }: IconProps) {
  return <ArrowRight size={size} {...iconProps} />;
}

export function CaretRightIcon({ size = 14 }: IconProps) {
  return <CaretRight size={size} {...iconProps} />;
}

export function SettingsIcon({ size = 16 }: IconProps) {
  return <Gear size={size} {...iconProps} />;
}

export function TrashIcon({ size = 14 }: IconProps) {
  return <Trash size={size} {...iconProps} />;
}

export function WarningIcon({ size = 16 }: IconProps) {
  return <Warning size={size} weight="fill" aria-hidden />;
}

export function CheckIcon({ size = 16 }: IconProps) {
  return <Check size={size} weight="bold" aria-hidden />;
}

export function PdfIcon({ size = 14 }: IconProps) {
  return <FilePdf size={size} {...iconProps} />;
}

export function EyeIcon({ size = 16 }: IconProps) {
  return <Eye size={size} {...iconProps} />;
}

export function EyeSlashIcon({ size = 16 }: IconProps) {
  return <EyeSlash size={size} {...iconProps} />;
}

export function ExternalLinkIcon({ size = 13 }: IconProps) {
  return <ArrowSquareOut size={size} {...iconProps} />;
}
