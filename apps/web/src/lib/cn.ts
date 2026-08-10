type ClassValue = string | false | null | undefined;

/**
 * Joins class names, dropping falsy values.
 *
 * Deliberately not `clsx` + `tailwind-merge`: this codebase composes classes
 * rather than overriding them, so conflict resolution would be dead weight in
 * a demo bundle.
 */
export function cn(...values: ClassValue[]): string {
  return values.filter(Boolean).join(" ");
}
