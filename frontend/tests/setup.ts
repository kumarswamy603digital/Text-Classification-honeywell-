import '@testing-library/jest-dom/vitest'
import { vi } from 'vitest'

// Recharts' ResponsiveContainer needs ResizeObserver, missing in jsdom
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver = vi.fn(
  () => new ResizeObserverStub(),
) as unknown as typeof ResizeObserver

// Minimal crypto.randomUUID for jsdom
if (!globalThis.crypto) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ;(globalThis as any).crypto = { randomUUID: () => 'test-uuid' }
}
