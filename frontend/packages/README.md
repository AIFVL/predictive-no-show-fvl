# Frontend packages

Estructura modular del cliente web.

```
frontend/
├── packages/
│   └── shared/          # @fvl/shared — constantes, estados, fechas, URLs API
├── src/
│   ├── hooks/           # Lógica de datos (useAppointments)
│   ├── lib/             # Predicciones y doble verificación
│   ├── App.jsx          # Composición de la UI
│   └── main.jsx
└── vite.config.js       # Alias @fvl/shared → packages/shared/src
```

## @fvl/shared

- `appointmentStatus.js` — estados Asistida, Asistirá, No asistirá, No asistió
- `dateWindow.js` — ventana de fechas (8 / 15 / 30 días)
- `api.js` — endpoints del backend con parámetro `days`

## Desarrollo

```bash
cd frontend
npm install
npm run dev
```

Variable opcional: `VITE_API_BASE_URL` (default `http://localhost:8000`).
