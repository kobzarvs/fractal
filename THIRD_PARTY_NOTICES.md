# Происхождение формул и эталона

Поведение Burning Ship, координаты маршрутов, алгоритмы perturbation/BLA, кольцевая раскладка и палитра изучены по публичной сборке https://newton-fractal.pages.dev/ на 2026-09-24:

- https://newton-fractal.pages.dev/assets/index-Dp_pt8ok.js
- https://newton-fractal.pages.dev/assets/reference-worker-xqHYK19O.js

`src/compute/reference-js.ts`, `src/tours.ts` и математические части `src/gpu` адаптируют эту сборку. Rust-арифметика, ABI, интеграция worker, интерфейс и тесты реализованы в этом проекте. Публичный URL сам по себе не задаёт лицензию; этот документ фиксирует происхождение и не предоставляет дополнительных прав на исходную реализацию.

Rust crate `libm` распространяется по MIT/Apache-2.0. Прочие зависимости и их версии зафиксированы в Cargo.lock и package-lock.json; лицензии остаются у соответствующих авторов.
