export type FrameRatePhase = 'starting' | 'failed' | 'lost' | 'hidden' | 'benchmark'
  | 'reference' | 'preparing' | 'stale' | 'idle' | 'active';
const messages: Record<Exclude<FrameRatePhase, 'idle' | 'active'>, string> = {
  starting: 'Загрузка движка…', failed: 'Ошибка движка · можно переключить',
  lost: 'Восстановление GPU…', hidden: 'Вкладка скрыта', benchmark: 'Идёт сравнение движков…',
  reference: 'Расчёт опорной орбиты…', preparing: 'Подготовка кэша…',
  stale: 'Нет свежих данных от рендера',
};

/** GPU completion estimates are not presentation timestamps. Never fall back
 * to submission cadence when confirmation or a fresh worker state is absent. */
export function gpuFrameRateDisplay(sample: { fps: number | null; pendingFrames: number }, phase: FrameRatePhase) {
  if (phase !== 'idle' && phase !== 'active') return { value: '—', detail: messages[phase] };
  if (phase === 'idle') return sample.pendingFrames > 0
    ? { value: '—', detail: 'Ожидание подтверждения GPU…' }
    : { value: '0', detail: 'Кадр готов · рендер приостановлен' };
  if (sample.fps === null || !Number.isFinite(sample.fps) || sample.fps < 0)
    return { value: '—', detail: sample.pendingFrames > 0 ? 'Ожидание подтверждения GPU…' : 'Нет подтверждённого замера GPU' };
  return { value: sample.fps.toFixed(1).replace(/\.0$/, ''), detail: 'Завершённые GPU кадры · оценка' };
}
