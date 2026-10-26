# Deployment Guide

## Local Containers

```bash
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env
docker compose up --build
```

## Cloud Readiness

- Backend is containerized with FastAPI and can run behind Nginx on AWS ECS, EKS, EC2, or GCP Cloud Run.
- Frontend is built into static assets suitable for S3 + CloudFront or container deployment.
- MongoDB can be replaced with MongoDB Atlas for managed scaling.
- Model artifacts can be stored in object storage and mounted during deployment.

