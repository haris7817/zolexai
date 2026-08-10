import type { Metadata } from "next";
import Link from "next/link";
import { AuthShell, AuthField } from "@/components/auth/AuthShell";

export const metadata: Metadata = { title: "Create account" };

export default function RegisterPage() {
  return (
    <AuthShell
      title="Start creating"
      subtitle="One plan, every tool. Cancel any time."
      submitLabel="Create Account"
      fields={
        <>
          <AuthField
            id="register-name"
            label="Full name"
            placeholder="Maya Adler"
          />
          <AuthField
            id="register-email"
            label="Email"
            type="email"
            placeholder="you@studio.com"
          />
          <AuthField
            id="register-password"
            label="Password"
            type="password"
            placeholder="At least 10 characters"
          />
          <p className="text-zx-text-muted mt-0 mb-5 text-[11.5px] leading-[1.5]">
            By creating an account you agree to the Terms and Privacy Policy.
          </p>
        </>
      }
      footer={
        <>
          Already have an account?{" "}
          <Link href="/login" className="font-bold">
            Sign in
          </Link>
        </>
      }
    />
  );
}
