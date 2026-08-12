import {
  Home,
  Sparkles,
  Image as ImageIcon,
  Repeat,
  Music,
  Clapperboard,
  LayoutGrid,
  History,
  Folder,
  CreditCard,
  Settings2,
  Download,
  RotateCcw,
  Copy,
  Share,
  Upload,
  ChevronDown,
  ChevronRight,
  ChevronLeft,
  SlidersHorizontal,
  LogOut,
  Menu,
  X,
  Search,
  Play,
  Pause,
  Trash2,
  Check,
  Plus,
  User,
  Bell,
  Shield,
  Palette,
  Wand2,
  FileVideo,
  FileAudio,
  FileImage,
  ArrowRight,
  ArrowUpRight,
  MoreHorizontal,
  Filter,
  Clock,
  CircleAlert,
  Loader,
  HelpCircle,
  Mail,
  Lock,
  Eye,
  EyeOff,
  type LucideProps,
} from "lucide-react";

/**
 * ===========================================================================
 * ZolexAI icon system — PREUI-02
 * ===========================================================================
 *
 * ONE icon system across every screen.
 *
 * The approved Video Workspace hand-drew 22 SVGs whose paths are Lucide's;
 * they are mapped here to `lucide-react` so the set is consistent, tree-shaken
 * and extendable.
 *
 * Landing and Creator Dashboard originally used Unicode glyphs as product
 * icons (✦ ◈ ⟲ ⇢ ♪ ▶ ⊞ ◫ ▤ ◇ ⚙ ⌂ ⏻). Those are removed: glyph coverage varies
 * by OS and font, so `⏻ ◫ ▤ ⇢` fall back inconsistently on Windows and Android
 * — the same build looks broken on some client machines. They also cannot
 * inherit stroke weight, so they never optically match a 1.8px-stroke SVG.
 * See ADR 0001 §3.
 *
 * Stroke width 1.8 matches the approved design exactly.
 */

/**
 * The one glyph with no Lucide equivalent — the "extend" mark from the
 * approved Workspace, kept at matching stroke weight.
 */
function ExtendIcon(props: LucideProps) {
  const { size = 24, ...rest } = props;
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...rest}
    >
      <path d="M2 12h17" />
      <path d="m15 8 4 4-4 4" />
      <path d="M22 5v14" />
    </svg>
  );
}

export const ICONS = {
  // Navigation
  home: Home,
  grid: LayoutGrid,
  history: History,
  folder: Folder,
  card: CreditCard,
  settings: Settings2,

  // Workflows — keys match WorkflowIconName
  sparkles: Sparkles,
  image: ImageIcon,
  repeat: Repeat,
  extend: ExtendIcon,
  music: Music,
  clapper: Clapperboard,

  // Result actions
  download: Download,
  reuse: RotateCcw,
  copy: Copy,
  share: Share,

  // Controls
  upload: Upload,
  chevron: ChevronDown,
  chevronRight: ChevronRight,
  chevronLeft: ChevronLeft,
  sliders: SlidersHorizontal,
  menu: Menu,
  close: X,
  search: Search,
  filter: Filter,
  clock: Clock,
  play: Play,
  pause: Pause,
  trash: Trash2,
  check: Check,
  plus: Plus,
  more: MoreHorizontal,
  arrowRight: ArrowRight,
  arrowUpRight: ArrowUpRight,

  // Account / settings
  logout: LogOut,
  user: User,
  bell: Bell,
  shield: Shield,
  palette: Palette,
  wand: Wand2,
  help: HelpCircle,
  mail: Mail,
  lock: Lock,
  eye: Eye,
  eyeOff: EyeOff,

  // Media kinds
  video: FileVideo,
  audio: FileAudio,
  picture: FileImage,

  // Status
  alert: CircleAlert,
  loader: Loader,
} as const;

export type IconName = keyof typeof ICONS;

export interface IconProps extends Omit<LucideProps, "ref"> {
  name: IconName;
  size?: number;
}

/**
 * Renders a ZolexAI icon. Decorative by default (`aria-hidden`) because icons
 * here always sit beside a text label; pass `aria-label` and `aria-hidden={false}`
 * for the rare standalone case.
 */
export function Icon({ name, size = 16, ...rest }: IconProps) {
  const Component = ICONS[name];
  return (
    <Component
      size={size}
      strokeWidth={1.8}
      aria-hidden="true"
      className="shrink-0"
      {...rest}
    />
  );
}
