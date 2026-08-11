# ZolexAI frontend.
#
#   docker build -f apps/web/Dockerfile .        (context = repo root)
#
# Context is the repo root because the build needs two things outside apps/web:
# the `@zolexai/workflow-contracts` workspace package, and `workflow-definitions/`
# which the landing page and app shell read at render time.

FROM node:20-alpine AS deps
WORKDIR /repo

# Manifests only, so a source change does not reinstall node_modules.
COPY package.json package-lock.json ./
COPY apps/web/package.json ./apps/web/
COPY packages/workflow-contracts/package.json ./packages/workflow-contracts/
RUN npm ci --workspaces --include-workspace-root


FROM node:20-alpine AS builder
WORKDIR /repo

COPY --from=deps /repo/node_modules ./node_modules
COPY --from=deps /repo/apps/web/node_modules ./apps/web/node_modules
COPY package.json package-lock.json ./
COPY packages ./packages
COPY apps/web ./apps/web
COPY workflow-definitions ./workflow-definitions

# The app has no static assets yet, so `apps/web/public` does not exist in the
# repository and Docker cannot COPY a missing path — the runner stage failed
# with "/repo/apps/web/public: not found". Creating it here keeps that COPY
# unconditional, and the directory becomes real the moment a favicon or an OG
# image is added, with no Dockerfile change needed.
RUN mkdir -p /repo/apps/web/public

# Baked into the bundle at build time, so it must be correct for the target
# environment — a NEXT_PUBLIC_* value cannot be changed at runtime.
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL} \
    NEXT_STANDALONE=1 \
    NEXT_TELEMETRY_DISABLED=1

WORKDIR /repo/apps/web
RUN npm run build


FROM node:20-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0

# `standalone` bundles only the files the server actually needs — a few MB
# instead of the whole workspace's node_modules.
COPY --from=builder /repo/apps/web/.next/standalone ./
COPY --from=builder /repo/apps/web/.next/static ./apps/web/.next/static
COPY --from=builder /repo/apps/web/public ./apps/web/public

# Read at request time by the landing page and app shell, so they must ship.
COPY --from=builder /repo/workflow-definitions ./workflow-definitions
ENV WORKFLOW_DEFINITIONS_DIR=/app/workflow-definitions

RUN addgroup -g 10003 zolexai && adduser -u 10003 -G zolexai -D zolexai
USER zolexai

EXPOSE 3000

CMD ["node", "apps/web/server.js"]
