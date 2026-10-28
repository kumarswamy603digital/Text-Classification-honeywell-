/** Shared accessible UI primitives (Button, Card, Badge, Select, states). */
import {
  autoUpdate,
  flip,
  offset,
  useDismiss,
  useFloating,
  useInteractions,
  useListNavigation,
  useRole,
} from '@floating-ui/react'
import * as React from 'react'
import { cn } from '../../utils/format'

// ---------- Card ----------
export function Card({
  className,
  title,
  subtitle,
  actions,
  children,
}: {
  className?: string
  title?: React.ReactNode
  subtitle?: React.ReactNode
  actions?: React.ReactNode
  children: React.ReactNode
}) {
  return (
    <section
      className={cn(
        'rounded-xl border border-line bg-panel p-5 shadow-lg shadow-black/20',
        className,
      )}
    >
      {(title || actions) && (
        <header className="mb-4 flex items-start justify-between gap-3">
          <div>
            {title && <h2 className="text-base font-semibold tracking-tight">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-sm text-muted">{subtitle}</p>}
          </div>
          {actions}
        </header>
      )}
      {children}
    </section>
  )
}

// ---------- Button ----------
type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger'
const buttonVariants: Record<ButtonVariant, string> = {
  primary:
    'bg-accent text-slate-950 hover:bg-sky-300 active:bg-sky-400 font-semibold disabled:bg-slate-600 disabled:text-slate-300',
  secondary: 'bg-panel2 border border-line text-ink hover:border-accent/60 hover:bg-slate-700/40',
  ghost: 'text-muted hover:text-ink hover:bg-panel2',
  danger: 'bg-bad/15 text-bad border border-bad/40 hover:bg-bad/25',
}

export function Button({
  variant = 'primary',
  className,
  loading,
  children,
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: ButtonVariant
  loading?: boolean
}) {
  return (
    <button
      className={cn(
        'inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2 text-sm transition-colors',
        'disabled:cursor-not-allowed disabled:opacity-60',
        buttonVariants[variant],
        className,
      )}
      disabled={props.disabled || loading}
      {...props}
    >
      {loading && (
        <span
          aria-hidden
          className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent"
        />
      )}
      {children}
    </button>
  )
}

// ---------- Badge ----------
export function Badge({
  tone = 'neutral',
  className,
  children,
}: {
  tone?: 'neutral' | 'success' | 'danger' | 'info' | 'warn'
  className?: string
  children: React.ReactNode
}) {
  const tones = {
    neutral: 'bg-slate-500/15 text-slate-300 border-slate-500/30',
    success: 'bg-emerald-500/15 text-good border-emerald-500/30',
    danger: 'bg-red-500/15 text-bad border-red-500/30',
    info: 'bg-sky-500/15 text-accent border-sky-500/30',
    warn: 'bg-amber-500/15 text-warn border-amber-500/30',
  }
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-medium',
        tones[tone],
        className,
      )}
    >
      {children}
    </span>
  )
}

// ---------- Accessible Select (roving focus, ARIA listbox, Esc/outside dismiss) ----------
export interface SelectOption {
  value: string
  label: string
  description?: string
}

export function Select({
  options,
  value,
  onChange,
  label,
  className,
  disabled,
}: {
  options: SelectOption[]
  value: string
  onChange: (v: string) => void
  label: string
  className?: string
  disabled?: boolean
}) {
  const [open, setOpen] = React.useState(false)
  const [activeIndex, setActiveIndex] = React.useState<number | null>(null)
  const listRef = React.useRef<Array<HTMLElement | null>>([])
  const selectedIndex = Math.max(
    0,
    options.findIndex((o) => o.value === value),
  )

  const { refs, floatingStyles, context } = useFloating({
    open,
    onOpenChange: setOpen,
    placement: 'bottom-start',
    whileElementsMounted: autoUpdate,
    middleware: [offset(6), flip()],
  })
  const dismiss = useDismiss(context, { enabled: open })
  const role = useRole(context, { role: 'select' })
  const listNavigation = useListNavigation(context, {
    enabled: open,
    listRef,
    activeIndex,
    onNavigate: setActiveIndex,
  })
  const { getReferenceProps, getFloatingProps, getItemProps } = useInteractions([
    role,
    dismiss,
    listNavigation,
  ])

  const current = options[selectedIndex]

  return (
    <div className={cn('relative', className)}>
      <span id={`${label}-label`} className="mb-1.5 block text-xs font-medium uppercase tracking-wider text-muted">
        {label}
      </span>
      <button
        ref={refs.setReference}
        aria-labelledby={`${label}-label`}
        aria-haspopup="listbox"
        aria-expanded={open}
        {...getReferenceProps({
          onClick: () => !disabled && setOpen((o) => !o),
          disabled,
          className: cn(
            'flex w-full items-center justify-between gap-2 rounded-lg border border-line bg-panel2 px-3 py-2 text-sm',
            'hover:border-accent/50 disabled:opacity-50',
          ),
        })}
      >
        <span className="truncate">{current ? current.label : 'Select…'}</span>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden className="text-muted">
          <path d="m6 9 6 6 6-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
        </svg>
      </button>
      {open && (
        <ul
          ref={refs.setFloating}
          {...getFloatingProps({
            style: floatingStyles,
            className: cn(
              'z-50 min-w-[16rem] overflow-auto rounded-lg border border-line bg-panel2 py-1 shadow-xl',
            ),
          })}
          role="listbox"
          aria-labelledby={`${label}-label`}
        >
          {options.map((opt, i) => (
            <li key={opt.value}>
              <button
                ref={(n) => {
                  listRef.current[i] = n
                }}
                {...getItemProps({
                  onClick: () => {
                    onChange(opt.value)
                    setOpen(false)
                  },
                  role: 'option',
                  'aria-selected': opt.value === value,
                  tabIndex: 0,
                  className: cn(
                    'block w-full px-3 py-2 text-left text-sm hover:bg-accent/10',
                    i === activeIndex && 'bg-accent/15',
                    opt.value === value && 'font-medium text-accent',
                  ),
                })}
              >
                {opt.label}
                {opt.description && (
                  <span className="block text-xs text-muted">{opt.description}</span>
                )}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
