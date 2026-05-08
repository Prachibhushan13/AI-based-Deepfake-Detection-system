# API Documentation

Base URL: `http://localhost:8000/api/v1`

## Authentication

- `POST /auth/signup`
- `POST /auth/login`

## Prediction

- `POST /upload-video`
- `POST /predict`
- `GET /history`
- `GET /report/{id}`
- `DELETE /history/{id}`

## Admin

- `GET /admin/stats`

Authentication uses Bearer JWT tokens in the `Authorization` header.

