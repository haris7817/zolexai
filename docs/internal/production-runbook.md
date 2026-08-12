# ZOLEXAI

Production Deployment Runbook

**Milestone 1 Production Infrastructure & Go-Live Procedure**

Prepared: 12 August 2026

> **Purpose:** Internal deployment and recovery guide for the ZolexAI production environment. It records the server layout, Docker stack, CloudPanel routing, DNS/SSL setup, deployment commands, validation tests, security controls, rollback steps, and known M1 limitations.

> **Security:** This runbook intentionally contains no passwords, private SSH keys, API tokens, database passwords, storage secrets, or production .env values. Never paste those secrets into tickets, chat, GitHub, or this document.

# Contents

1. [Deployment overview](#1-deployment-overview)
2. [Production architecture](#2-production-architecture)
3. [Server inventory and boundaries](#3-server-inventory-and-boundaries)
4. [Pre-deployment safety and backup](#4-pre-deployment-safety-and-backup)
5. [Existing EMB25AI cleanup](#5-existing-emb25ai-cleanup)
6. [Docker and deployment user setup](#6-docker-and-deployment-user-setup)
7. [GitHub deploy-key and repository clone](#7-github-deploy-key-and-repository-clone)
8. [Production environment configuration](#8-production-environment-configuration)
9. [Production Docker Compose stack](#9-production-docker-compose-stack)
10. [Build, migrate, and start services](#10-build-migrate-and-start-services)
11. [Deployment issues discovered and fixes](#11-deployment-issues-discovered-and-fixes)
12. [DNS and CloudPanel configuration](#12-dns-and-cloudpanel-configuration)
13. [Main domain reverse proxy and API routing](#13-main-domain-reverse-proxy-and-api-routing)
14. [SSL for main domain](#14-ssl-for-main-domain)
15. [Storage subdomain / MinIO reverse proxy](#15-storage-subdomain--minio-reverse-proxy)
16. [SSL for storage](#16-ssl-for-storage)
17. [Security validation](#17-security-validation)
18. [Functional go-live testing](#18-functional-go-live-testing)
19. [Routine update deployment](#19-routine-update-deployment)
20. [Restart, reboot and recovery](#20-restart-reboot-and-recovery)
21. [Rollback procedure](#21-rollback-procedure)
22. [Troubleshooting guide](#22-troubleshooting-guide)
23. [M1 production status and limitations](#23-m1-production-status-and-limitations)
24. [Final checklist](#24-final-checklist)

# 1. Deployment overview

ZolexAI is deployed as a containerized multi-service application on a Hostinger VPS. CloudPanel and the host Nginx installation remain the public edge. Docker services are bound only to loopback or private Docker networking; public traffic enters through Nginx on HTTPS.

| **Item**              | **Production value**                                          |
|-----------------------|---------------------------------------------------------------|
| Primary domain        | https://zolexai.com                                           |
| WWW                   | https://www.zolexai.com -> redirects to https://zolexai.com/ |
| Storage domain        | https://storage.zolexai.com                                   |
| VPS                   | Hostinger KVM8 - Ubuntu 24.04                                 |
| Public VPS IP         | 76.13.27.242                                                  |
| Application directory | /opt/zolexai                                                  |
| Repository            | Private GitHub repository: haris7817/zolexai                  |
| Edge                  | CloudPanel + host Nginx                                       |
| Runtime               | Docker + Docker Compose                                       |
| Current worker        | Mock worker for M1; real GPU/model worker belongs to M2       |

## Deployment goals

- Keep CloudPanel, Nginx, MySQL and existing host services intact.

- Do not expose PostgreSQL, Docker Redis, worker endpoints, or MinIO console publicly.

- Serve the website and public API from the same origin to simplify frontend configuration.

- Serve browser uploads from a dedicated HTTPS storage subdomain using MinIO presigned URLs.

- Make the application restartable and independently scalable without rebuilding the architecture.

# 2. Production architecture

```bash
Internet
|
+--> https://zolexai.com/
| CloudPanel/Nginx -> 127.0.0.1:3100 -> Next.js web
|
+--> https://zolexai.com/api/v1/*
| CloudPanel/Nginx -> 127.0.0.1:8100 -> FastAPI
|
+--> https://storage.zolexai.com/*
CloudPanel/Nginx -> 127.0.0.1:9000 -> MinIO S3 API

Docker private network:
web -> api -> PostgreSQL / Redis / MinIO
^
|
worker
```

| **Service** | **Container role**           | **Host exposure**                                           |
|-------------|------------------------------|-------------------------------------------------------------|
| web         | Next.js frontend             | 127.0.0.1:3100 -> container 3000                           |
| api         | FastAPI backend              | 127.0.0.1:8100 -> container 8000                           |
| worker      | Generation worker            | No public/host port                                         |
| postgres    | Application database         | Private Docker network only, 5432                           |
| redis       | Queue/coordination           | Private Docker network only, 6379                           |
| minio       | S3-compatible object storage | 127.0.0.1:9000 -> container 9000; console 9001 not exposed |

> **Important:** The VPS already has a host-level Redis on 127.0.0.1:6379. ZolexAI uses a separate Redis container on the Docker network. Do not remove or reconfigure the host Redis as part of ZolexAI deployment.

# 3. Server inventory and boundaries

| **Component**                | **Current state / rule**                             |
|------------------------------|------------------------------------------------------|
| VPS hostname                 | srv1564171.hstgr.cloud                               |
| OS                           | Ubuntu 24.04                                         |
| Resources                    | KVM8; 8 vCPU, approx. 31 GB RAM, approx. 387 GB disk |
| CloudPanel                   | Already installed; administrative interface on 8443  |
| Host Nginx                   | Already active; owns ports 80/443                    |
| clp-nginx                    | Already active; do not replace                       |
| MySQL/Percona                | Existing host service; unrelated to ZolexAI          |
| UFW                          | Active; public web ports are already managed         |
| Docker                       | Installed from official Docker repository            |
| Application Linux user       | zolexai                                              |
| CloudPanel web-site user     | zolexaiweb                                           |
| CloudPanel storage-site user | zolexaistorage                                       |

> **Boundary rule:** Do not install an Nginx container that binds host ports 80/443. CloudPanel/host Nginx is the edge proxy and certificate manager.

# 4. Pre-deployment safety and backup

Before modifying an existing VPS, inventory active services and preserve the previous application. The deployment was performed without reinstalling the operating system or replacing CloudPanel.

```bash
systemctl is-active nginx clp-nginx mysql redis
ss -tulpn
df -h
free -h
```

| **Backup item**           | **Location / note**                    |
|---------------------------|----------------------------------------|
| Legacy application backup | /root/pre-zolexai-backup-2026-08-11/   |
| Legacy home archive       | emb25ai-home.tar.gz (approx. 171 MB)   |
| Legacy Nginx vhost        | emb25ai.com.conf / nginx-disabled copy |
| Legacy TLS materials      | Certificate/key copied to backup       |
| PM2 state                 | dump.pm2 copied to backup              |

> **Do not delete the backup yet:** Even though the client gave permission to remove the old EMB25AI application, retain the backup until ZolexAI is stable and a later retention decision is made.

# 5. Existing EMB25AI cleanup

The previous EMB25AI Node/PM2 application was already broken because expected frontend files were missing. Cleanup was intentionally limited to freeing the port and disabling its public vhost while preserving backup data.

1.  Inspect PM2 and identify the legacy app/process using port 3000.

```bash
pm2 list
ss -ltnp | grep :3000
```

2.  Stop and delete the legacy PM2 apps, then persist PM2 state if appropriate.

```bash
pm2 stop emb255
pm2 delete emb255
# Also remove any dead legacy emb entries after verifying they belong to the old app.
pm2 save
```

3.  Move the old enabled Nginx vhost to the backup instead of permanently deleting it, then validate and reload Nginx.

```bash
nginx -t
systemctl reload nginx
```

> **CloudPanel UI note:** The old emb25ai.com site may still appear in CloudPanel even after its PM2 process and enabled Nginx vhost were disabled. Do not modify that entry during ZolexAI go-live unless you intentionally plan a separate cleanup.

# 6. Docker and deployment user setup

## 6.1 Docker

Docker Engine and Docker Compose were installed from Docker's official repository. Confirm the installation:

```bash
docker --version
docker compose version
docker run --rm hello-world
```

Observed production versions during deployment were Docker 29.7.2 and Docker Compose v5.4.0.

## 6.2 Deployment user and directory

```bash
id zolexai
ls -ld /opt/zolexai
```

The zolexai Linux user owns /opt/zolexai and is a member of the docker group. Run Git operations as this user to avoid ownership and Git safe-directory problems.

> **Git rule:** Use runuser -u zolexai -- git ... when operating from a root shell. Running Git as root against /opt/zolexai can trigger "dubious ownership" warnings.

# 7. GitHub deploy-key and repository clone

A read-only SSH deploy key is used so the production server can pull the private repository without storing a personal GitHub password/token.

```bash
# Key path used on the VPS
/home/zolexai/.ssh/github_zolexai

# Public key is added in GitHub repository settings as a read-only Deploy Key.
# Never copy the private key into chat, tickets, or documentation.
```

```bash
runuser -u zolexai -- ssh -T git@github.com
# Expected: successful authentication message for haris7817/zolexai
```

```bash
cd /opt
# Clone with the configured deploy key / SSH config.
runuser -u zolexai -- git clone git@github.com:haris7817/zolexai.git /opt/zolexai
cd /opt/zolexai
runuser -u zolexai -- git status
```

# 8. Production environment configuration

Production configuration is stored in /opt/zolexai/.env with permission mode 600 and ownership zolexai:zolexai. The real values must never be committed to Git.

```bash
chmod 600 /opt/zolexai/.env
chown zolexai:zolexai /opt/zolexai/.env
ls -l /opt/zolexai/.env
```

| **Variable / group**    | **Production setting / rule**                |
|-------------------------|----------------------------------------------|
| APP_ENV                 | production                                   |
| APP_URL                 | https://zolexai.com                          |
| API_URL                 | https://zolexai.com                          |
| LOG_FORMAT              | json                                         |
| PostgreSQL host         | postgres:5432 on Docker network              |
| Redis host              | redis:6379 on Docker network                 |
| STORAGE_PROVIDER        | minio                                        |
| STORAGE_ENDPOINT        | http://minio:9000                            |
| STORAGE_PUBLIC_ENDPOINT | https://storage.zolexai.com                  |
| STORAGE_BUCKET          | zolexai-prod                                 |
| CORS_ORIGINS            | https://zolexai.com,https://www.zolexai.com  |
| API_PORT                | 8100                                         |
| WEB_PORT                | 3100                                         |
| NEXT_PUBLIC_API_URL     | empty; frontend uses same-origin /api/v1/... |
| API_BASE_URL            | http://api:8000 for internal/container use   |
| Worker runtime          | mock for M1                                  |

> **Required secrets:** Generate strong unique values for POSTGRES_PASSWORD, STORAGE_ACCESS_KEY, STORAGE_SECRET_KEY and WORKER_API_TOKEN (at least 32 characters). Keep them only in the protected production .env / secret store.

# 9. Production Docker Compose stack

The production Compose definition is stored at:

```bash
/opt/zolexai/infrastructure/compose/docker-compose.prod.yml
```

The production file is intentionally separate from local-development Compose configuration. It defines six services and persistent volumes for PostgreSQL and MinIO.

| **Service** | **Essential production behavior**                                                  |
|-------------|------------------------------------------------------------------------------------|
| postgres    | postgres:16-alpine; no host port; persistent postgres_data volume                  |
| redis       | redis:7-alpine; private network; no host persistence currently                     |
| minio       | Pinned MinIO release; loopback 127.0.0.1:9000 only; persistent minio_data volume   |
| api         | Build from repository; loopback 127.0.0.1:8100 -> 8000                            |
| worker      | Build from repository; no host ports; uses internal API and worker token           |
| web         | Build from repository; loopback 127.0.0.1:3100 -> 3000; NEXT_PUBLIC_API_URL empty |

```bash
cd /opt/zolexai
docker compose --env-file .env -f infrastructure/compose/docker-compose.prod.yml config --quiet
```

> **Redis kernel setting:** vm.overcommit_memory=1 was enabled and persisted in /etc/sysctl.d/99-redis-overcommit.conf to avoid Redis memory-allocation warnings.

# 10. Build, migrate, and start services

4.  Pull the desired Git revision as the zolexai user.

```bash
cd /opt/zolexai
runuser -u zolexai -- git fetch --all --tags
runuser -u zolexai -- git status
runuser -u zolexai -- git pull --ff-only
```

5.  Validate Compose configuration before starting anything.

```bash
docker compose --env-file .env -f infrastructure/compose/docker-compose.prod.yml config --quiet
```

6.  Build application images.

```bash
docker compose --env-file .env -f infrastructure/compose/docker-compose.prod.yml build api worker web
```

7.  Start infrastructure services first.

```bash
docker compose --env-file .env -f infrastructure/compose/docker-compose.prod.yml up -d postgres redis minio
```

8.  Run the database migration. Use the repository's Alembic command/container entrypoint as currently defined.

```bash
# Example pattern used during deployment:
docker compose --env-file .env -f infrastructure/compose/docker-compose.prod.yml run --rm api alembic upgrade head
```

9.  Start the API, worker and web services.

```bash
docker compose --env-file .env -f infrastructure/compose/docker-compose.prod.yml up -d api worker web
```

10. Inspect status and logs.

```bash
docker compose --env-file .env -f infrastructure/compose/docker-compose.prod.yml ps
docker compose --env-file .env -f infrastructure/compose/docker-compose.prod.yml logs --tail=100 api worker web
```

## 10.1 Expected local health checks

```bash
curl -i http://127.0.0.1:8100/api/v1/health/live
curl -i http://127.0.0.1:8100/api/v1/health
curl -I http://127.0.0.1:3100
```

The full API health response should report database, redis, storage and workflows as true.

# 11. Deployment issues discovered and fixes

| **Problem**                               | **Cause**                                                         | **Fix now present**                                                                                                       |
|-------------------------------------------|-------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| Alembic IndexError in REPO_ROOT           | Code assumed a fixed parent depth that differs inside Docker      | API and worker now resolve repository root without relying on parents\[4\]; deployment layout tests added.                |
| ModuleNotFoundError: app.models           | .gitignore rule models/ ignored apps/api/app/models               | Added explicit tracking exceptions for apps/api/app/models/\*.py; removed accidentally tracked \_\_pycache\_\_/pyc files. |
| Missing lightningcss Linux native module  | Alpine production build needed linux-x64-musl optional dependency | Added lightningcss-linux-x64-musl compatible with the project version.                                                    |
| Missing Tailwind oxide native binding     | Tailwind CSS v4 build needed Alpine musl native package           | Added @tailwindcss/oxide-linux-x64-musl matching project version.                                                         |
| Web Docker runner COPY failed for /public | Next.js app had no public directory                               | Dockerfile creates /repo/apps/web/public before runner-stage COPY.                                                        |
| MinIO mc cors set unsupported             | Current MinIO build did not implement that CORS command path      | Set MINIO_API_CORS_ALLOW_ORIGIN to https://zolexai.com,https://www.zolexai.com and recreated MinIO.                       |
| Npm audit reported high vulnerabilities   | Potential forced dependency upgrades could break deployment       | Do not use npm audit fix --force during deployment. Handle dependency remediation separately with testing.                |

> **Repository state:** The production repository already includes the code fixes above. They are documented here mainly for diagnosis if a future branch/rebase reintroduces similar failures.

# 12. DNS and CloudPanel configuration

## 12.1 DNS records

| **Type** | **Name** | **Value**    | **TTL used** |
|----------|----------|--------------|--------------|
| A        | @        | 76.13.27.242 | 300          |
| CNAME    | www      | zolexai.com  | 300          |
| A        | storage  | 76.13.27.242 | 300          |

```bash
getent ahostsv4 zolexai.com
getent ahostsv4 www.zolexai.com
getent ahostsv4 storage.zolexai.com
```

## 12.2 CloudPanel sites

| **CloudPanel site** | **Type**      | **Reverse Proxy URL** | **Site user**  |
|---------------------|---------------|-----------------------|----------------|
| zolexai.com         | Reverse Proxy | http://127.0.0.1:3100 | zolexaiweb     |
| storage.zolexai.com | Reverse Proxy | http://127.0.0.1:9000 | zolexaistorage |

> **Do not use the Linux deployment user as the CloudPanel site user:** CloudPanel creates/manages its own site users. The existing Linux user zolexai caused "This value already exists", so the web site uses zolexaiweb and storage uses zolexaistorage.

# 13. Main domain reverse proxy and API routing

CloudPanel already generated the main reverse-proxy vhost for the web app. Do not overwrite the entire generated file. In CloudPanel > zolexai.com > Manage > Vhost, keep the generated content and ensure the following location blocks appear near the bottom, before the final location / block.

```bash
location ^~ /.well-known {
auth_basic off;
allow all;
try_files $uri @reverse_proxy;
}

location ^~ /api/v1/internal/ {
return 404;
}

location ^~ /api/v1/ {
proxy_pass http://127.0.0.1:8100;
proxy_http_version 1.1;

proxy_set_header Host $host;
proxy_set_header X-Real-IP $remote_addr;
proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
proxy_set_header X-Forwarded-Proto $scheme;
proxy_set_header X-Forwarded-Host $host;

proxy_buffering off;
proxy_cache off;

proxy_connect_timeout 60;
proxy_send_timeout 900;
proxy_read_timeout 900;
}

location / {
try_files $uri @reverse_proxy;
}
```

> **Why internal is blocked:** The public /api/v1/internal/ path returns 404 at Nginx. The worker does not need this public route; it reaches FastAPI through Docker networking at http://api:8000 with worker authentication.

> **Why buffering is off:** The generation events endpoint uses streaming/SSE behavior. Disabling proxy buffering prevents Nginx from unnecessarily holding streamed events.

# 14. SSL for main domain

In CloudPanel > zolexai.com > Manage > SSL/TLS, create and install a Let's Encrypt certificate containing:

```bash
zolexai.com
www.zolexai.com
```

```bash
curl -I https://zolexai.com
curl -i https://zolexai.com/api/v1/health
curl -I https://www.zolexai.com
nginx -t
```

Expected results: main domain HTTP 200; API health HTTP 200; www HTTP 301 redirect to https://zolexai.com/; Nginx syntax test successful.

> **OCSP stapling warning:** The deployed certificates may produce an Nginx warning such as "ssl_stapling ignored, no OCSP responder URL in the certificate". This is non-blocking when curl validates HTTPS without -k and nginx -t succeeds.

# 15. Storage subdomain / MinIO reverse proxy

The storage site reverse-proxies only the MinIO S3 API on port 9000. Do not expose MinIO console port 9001. In CloudPanel > storage.zolexai.com > Manage > Vhost, keep CloudPanel's generated structure and apply these production settings.

```bash
server {
# CloudPanel-generated listen / SSL directives remain unchanged.
server_name storage.zolexai.com;

# ... generated logging / HTTPS redirect ...

location @reverse_proxy {
proxy_pass {{reverse_proxy_url}};
proxy_http_version 1.1;
proxy_set_header X-Forwarded-Host $host;
proxy_set_header X-Forwarded-Server $host;
proxy_set_header X-Real-IP $remote_addr;
proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
proxy_set_header X-Forwarded-Proto $scheme;
proxy_set_header Host $http_host;
proxy_set_header Upgrade $http_upgrade;
proxy_set_header Connection "Upgrade";
proxy_ssl_server_name on;
proxy_ssl_name $host;
proxy_pass_request_headers on;
proxy_request_buffering off;
proxy_buffering off;
proxy_max_temp_file_size 0;
proxy_connect_timeout 900;
proxy_send_timeout 900;
proxy_read_timeout 900;
proxy_buffer_size 128k;
proxy_buffers 4 256k;
proxy_busy_buffers_size 256k;
proxy_temp_file_write_size 256k;
}

{{settings}}

client_max_body_size 1024m;
client_body_timeout 900s;

include /etc/nginx/global_settings;
add_header Cache-Control no-transform;
index index.html;

location ^~ /.well-known {
auth_basic off;
allow all;
try_files $uri @reverse_proxy;
}

location / {
try_files $uri @reverse_proxy;
}
}
```

> **Signed-URL requirement:** Preserve the exact Host header with proxy_set_header Host \$http_host;. SigV4-style presigned URLs can depend on the host used when signing.

> **Upload behavior:** proxy_request_buffering off and a 1024 MB client body limit prevent Nginx from unnecessarily buffering large browser uploads before forwarding them to MinIO.

# 16. SSL for storage

In CloudPanel > storage.zolexai.com > Manage > SSL/TLS, issue a Let's Encrypt certificate for only:

```bash
storage.zolexai.com
```

```bash
curl -i https://storage.zolexai.com/minio/health/live
nginx -t
```

Expected MinIO health result: HTTP 200 with an empty body. This confirms DNS, trusted TLS, Nginx proxying and MinIO availability.

# 17. Security validation

## 17.1 Public internal API block

```bash
curl -i https://zolexai.com/api/v1/internal/workers/register
```

Expected: HTTP 404 returned by Nginx.

## 17.2 Port exposure

```bash
ss -tulpn
```

| **Port / service** | **Expected exposure**                  |
|--------------------|----------------------------------------|
| 80/443             | Public host Nginx                      |
| 8443               | CloudPanel admin as configured by host |
| 3100               | 127.0.0.1 only                         |
| 8100               | 127.0.0.1 only                         |
| 9000               | 127.0.0.1 only                         |
| 9001               | Not exposed                            |
| PostgreSQL 5432    | Docker private only                    |
| Docker Redis 6379  | Docker private only                    |
| Host Redis 6379    | 127.0.0.1 only                         |

## 17.3 Secret hygiene

- Never print /opt/zolexai/.env in support chats or client messages.

- Never expose the private GitHub deploy key.

- Never publish WORKER_API_TOKEN, PostgreSQL password, MinIO access key or MinIO secret key.

- Keep the MinIO bucket private; use presigned URLs for browser uploads/downloads.

- Do not enable the MinIO console publicly.

# 18. Functional go-live testing

11. Open https://zolexai.com in a normal browser and verify no certificate warning.

12. Open the application and confirm workflows load from the backend.

13. Test Image-to-Video with a small JPG/PNG. The browser upload should use a presigned https://storage.zolexai.com/... URL, not localhost or a raw server port.

14. Submit a generation and confirm the M1 job reaches Completed through the mock worker.

15. Open Generation History / detail view and confirm the result is persisted after page refresh.

16. Test Video-to-Video with source video, and also with the optional reference image input.

17. Confirm Media Library shows uploaded/generated assets.

18. Confirm job progress/events update; inspect the browser Network tab if SSE appears stalled.

```bash
curl -i https://zolexai.com/api/v1/health
curl -i https://storage.zolexai.com/minio/health/live
curl -I https://www.zolexai.com
```

> **M1 test meaning:** A completed job currently validates website -> API -> database/Redis -> object storage -> worker/job lifecycle. It does not validate final GPU video quality because M1 uses the mock worker.

# 19. Routine update deployment

Use this sequence for normal application updates after the initial deployment. Review migrations and infrastructure changes before running them in production.

```bash
cd /opt/zolexai

# 1) Inspect current state
runuser -u zolexai -- git status
runuser -u zolexai -- git rev-parse --short HEAD

# 2) Pull reviewed code
runuser -u zolexai -- git pull --ff-only

# 3) Validate compose
docker compose --env-file .env -f infrastructure/compose/docker-compose.prod.yml config --quiet

# 4) Rebuild changed app images
docker compose --env-file .env -f infrastructure/compose/docker-compose.prod.yml build api worker web

# 5) Run migrations when the release contains schema changes
docker compose --env-file .env -f infrastructure/compose/docker-compose.prod.yml run --rm api alembic upgrade head

# 6) Recreate app services
docker compose --env-file .env -f infrastructure/compose/docker-compose.prod.yml up -d api worker web

# 7) Verify
docker compose --env-file .env -f infrastructure/compose/docker-compose.prod.yml ps
curl -i https://zolexai.com/api/v1/health
```

> **Recommended release discipline:** Record the previous Git commit before each deployment. Prefer tagged/reviewed releases. Do not make large dependency upgrades and infrastructure changes in the same emergency deployment.

# 20. Restart, reboot and recovery

## 20.1 Restart only ZolexAI

```bash
cd /opt/zolexai
docker compose --env-file .env -f infrastructure/compose/docker-compose.prod.yml restart api worker web
```

## 20.2 Full ZolexAI stack restart

```bash
cd /opt/zolexai
docker compose --env-file .env -f infrastructure/compose/docker-compose.prod.yml down
docker compose --env-file .env -f infrastructure/compose/docker-compose.prod.yml up -d
```

> **Persistence:** PostgreSQL and MinIO use named persistent volumes. Do not add -v to docker compose down unless you intentionally want to delete production data.

## 20.3 Host reboot validation

```bash
systemctl is-active nginx clp-nginx mysql redis
docker ps
nginx -t
curl -i https://zolexai.com/api/v1/health
curl -i https://storage.zolexai.com/minio/health/live
```

Compose services use restart policies so the application should recover after reboot; still perform the checks above.

# 21. Rollback procedure

Rollback should be deliberate. Application rollback is usually safer than rolling back database schema blindly.

19. Identify the previously known-good Git commit/tag and record the current failing commit.

```bash
cd /opt/zolexai
runuser -u zolexai -- git log --oneline -10
```

20. Check whether the failed release included an irreversible database migration. If yes, restore from a tested database backup rather than guessing a downgrade.

21. Checkout the known-good revision, rebuild, and recreate only the application services.

```bash
runuser -u zolexai -- git checkout <KNOWN_GOOD_COMMIT_OR_TAG>
docker compose --env-file .env -f infrastructure/compose/docker-compose.prod.yml build api worker web
docker compose --env-file .env -f infrastructure/compose/docker-compose.prod.yml up -d api worker web
```

22. Run health checks and inspect logs.

```bash
docker compose --env-file .env -f infrastructure/compose/docker-compose.prod.yml logs --tail=200 api worker web
curl -i https://zolexai.com/api/v1/health
```

> **Legacy application rollback:** A separate pre-ZolexAI backup exists under /root/pre-zolexai-backup-2026-08-11/. Restoring the old site should be considered only if explicitly required; do not overwrite the functioning ZolexAI CloudPanel/Nginx configuration without a plan.

# 22. Troubleshooting guide

| **Symptom**                               | **Check**                                           | **Likely action**                                                                                          |
|-------------------------------------------|-----------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| 502 Bad Gateway on main site              | curl 127.0.0.1:3100; docker compose ps/logs web     | Start/rebuild web; verify CloudPanel reverse proxy URL remains http://127.0.0.1:3100.                      |
| API /api/v1 returns frontend HTML/404     | Inspect zolexai.com Vhost location order            | Restore dedicated location ^~ /api/v1/ block before location /.                                            |
| API health fails database                 | docker compose ps/logs postgres api                 | Verify DATABASE_URL/password/container health; do not expose DB publicly.                                  |
| API health fails Redis                    | docker compose ps/logs redis api; kernel overcommit | Verify Docker Redis hostname redis and vm.overcommit_memory=1.                                             |
| API health fails storage                  | curl MinIO health locally; inspect minio logs       | Verify MinIO, bucket zolexai-prod and credentials/public endpoint.                                         |
| Browser upload fails / signature mismatch | Inspect upload host and Nginx Host header           | Use https://storage.zolexai.com presigned URL; preserve Host \$http_host; do not rewrite the signed query. |
| 413 Request Entity Too Large              | Storage Vhost body limit                            | Confirm client_max_body_size 1024m.                                                                        |
| Large upload stalls                       | Storage Vhost buffering/timeouts                    | Confirm proxy_request_buffering off, proxy_buffering off, 900s timeouts.                                   |
| SSE/progress appears delayed              | Main API proxy buffering                            | Confirm proxy_buffering off / proxy_cache off for /api/v1/.                                                |
| Internal worker route publicly reachable  | curl public internal path                           | Restore location ^~ /api/v1/internal/ { return 404; } before public API block.                             |
| nginx -t duplicate location error         | Vhost contains repeated blocks                      | Keep only one /.well-known and one location / block.                                                       |
| SSL curl works only with -k               | Certificate not trusted/installed                   | Install Let's Encrypt certificate in CloudPanel and retest without -k.                                     |
| ssl_stapling no OCSP URL warning          | nginx -t otherwise successful, curl trusted         | Non-blocking; do not treat as TLS failure by itself.                                                       |
| Git dubious ownership                     | Git was run as root in /opt/zolexai                 | Run Git as zolexai via runuser.                                                                            |
| Frontend Alpine native module missing     | Build log mentions lightningcss/oxide musl binding  | Confirm optional musl dependencies remain in package lock/package.json.                                    |
| Alembic cannot import app.models          | models directory ignored/missing                    | Confirm apps/api/app/models/\*.py tracked by Git and present in image.                                     |

# 23. M1 production status and limitations

| **Area**               | **M1 status**                                                                                      |
|------------------------|----------------------------------------------------------------------------------------------------|
| Frontend               | Live on zolexai.com; dark ZolexAI UI and workflow screens available                                |
| Backend                | FastAPI live behind same-origin /api/v1                                                            |
| Database               | PostgreSQL healthy and persistent                                                                  |
| Queue/coordination     | Redis healthy                                                                                      |
| Object storage         | Private MinIO bucket; public presigned endpoint via storage.zolexai.com                            |
| Worker                 | Mock runtime; job lifecycle operational                                                            |
| HTTPS                  | Trusted certificates on main/www and storage                                                       |
| Internal API           | Publicly blocked at Nginx                                                                          |
| Real AI/GPU generation | Not M1; Milestone 2                                                                                |
| Accounts/billing       | Not connected yet in preview build                                                                 |
| SEO                    | Current build contains noindex, nofollow                                                           |
| Brand logo             | Final client logo integration still needs visual work; current ZolexAI mark is placeholder/partial |

> **Do not overstate M1:** The platform foundation and end-to-end job/storage plumbing are live. Real LTX/GPU workflow quality, long-duration chaining, music generation improvements, subscriptions and final production branding are later work.

# 24. Final checklist

- [ ] Host Nginx / CloudPanel kept intact; no Docker service binds 80/443.
- [ ] Legacy EMB25AI backup retained.
- [ ] Docker and Compose installed and functional.
- [ ] zolexai deployment user owns /opt/zolexai and can use Docker.
- [ ] Read-only GitHub deploy key authenticates successfully.
- [ ] Production .env exists with mode 600 and is not committed.
- [ ] Compose config validates successfully.
- [ ] PostgreSQL, Redis and MinIO are healthy.
- [ ] Database migration is at the expected head.
- [ ] API health reports database=true, redis=true, storage=true, workflows=true.
- [ ] Worker is running (mock in M1).
- [ ] Web responds on 127.0.0.1:3100.
- [ ] DNS @, www and storage point to the VPS.
- [ ] zolexai.com HTTPS returns 200 without curl -k.
- [ ] www.zolexai.com redirects to zolexai.com.
- [ ] /api/v1 routes to FastAPI and SSE buffering is disabled.
- [ ] /api/v1/internal is blocked publicly with 404.
- [ ] storage.zolexai.com HTTPS MinIO health returns 200.
- [ ] Storage Vhost preserves \$http_host, disables request buffering and allows large uploads.
- [ ] PostgreSQL/Redis/MinIO console are not publicly exposed.
- [ ] Browser upload uses storage.zolexai.com presigned URL.
- [ ] M1 mock generation completes and persists in history/media library.
- [ ] Final logo/branding and real GPU model integration are tracked for later work.

# Quick command reference

```bash
cd /opt/zolexai
COMPOSE="docker compose --env-file .env -f infrastructure/compose/docker-compose.prod.yml"

$COMPOSE ps
$COMPOSE logs --tail=100 api worker web
curl -i https://zolexai.com/api/v1/health
curl -i https://storage.zolexai.com/minio/health/live
curl -i https://zolexai.com/api/v1/internal/workers/register
curl -I https://www.zolexai.com
nginx -t
```

> **Document maintenance:** Update this runbook whenever production ports, domains, Compose services, storage provider, worker runtime, deployment path, or CloudPanel routing changes. Never add live secrets.
