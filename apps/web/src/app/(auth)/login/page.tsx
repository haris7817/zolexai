import type { Metadata } from "next";
import Link from "next/link";
import { AuthShell, AuthField } from "@/components/auth/AuthShell";

export const metadata: Metadata = { title: "Sign in" };

export default function LoginPage() {
  return (
    <AuthShell
      title="Welcome back"
      subtitle="Sign in to pick up where you left off."
      submitLabel="Sign In"
      fields={
        <>
          <AuthField
            id="login-email"
            label="Email"
            type="email"
            placeholder="you@studio.com"
          />
          <AuthField
            id="login-password"
            label="Password"
            type="password"
            placeholder="••••••••"
          />
          <div className="mb-5 text-right">
            <span className="text-zx-text-muted text-[12.5px] font-semibold">
              Forgot password?
            </span>
          </div>
        </>
      }
      footer={
        <>
          New here?{" "}
          <Link href="/register" className="font-bold">
            Create an account
          </Link>
        </>
      }
    />
  );
}
