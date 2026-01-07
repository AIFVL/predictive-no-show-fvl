Frontend calendar (FullCalendar)

Files:
- `frontend/calendar/index.html` — static page that uses FullCalendar (CDN) and fetches `GET /appointments` from the backend.

How to run:
1. Make sure your FastAPI backend is running on `http://127.0.0.1:8000`.
2. Serve the `frontend/calendar` folder with a static file server. From the repo root you can use Python:

```bash
cd frontend/calendar
python -m http.server 5500
```

3. Open browser: `http://127.0.0.1:5500` — the calendar will load and call the backend endpoints.

Notes:
- CORS for the backend is already enabled (`allow_origins=['*']`).
- The calendar maps appointment `hour/day/month` to a date using the current year.
- Creating a new appointment calls `POST /appointments` and refreshes the calendar on success.
