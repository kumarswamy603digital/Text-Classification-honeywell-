/** App shell: sidebar navigation, API status pill, outlet for pages. */
import {
  BarChart3,
  Bot,
  Database,
  FileStack,
  Home,
  PenLine,
  BookOpen,
} from 'lucide-react'
import { NavLink, Outlet } from 'react-router-dom'
import { useApi } from '../../hooks/useApi'
import { api } from '../../api/client'
import { Badge } from '../ui/primitives'

const nav = [
  { to: '/', label: 'Dashboard', icon: Home, end: true },
  { to: '/studio', label: 'Classification Studio', icon: PenLine },
  { to: '/batch', label: 'Batch Classification', icon: FileStack },
  { to: '/compare', label: 'Model Comparison', icon: BarChart3 },
  { to: '/dataset', label: 'Dataset Explorer', icon: Database },
  { to: '/about', label: 'About & Docs', icon: BookOpen },
]

function ApiStatusPill() {
  const { data, loading, error } = useApi(() => api.ready())
  if (loading) return <Badge tone="neutral">checking API…</Badge>
  if (error) return <Badge tone="danger">API offline</Badge>
  if (data?.status === 'ready')
    return <Badge tone="success">API ready · {data.selected_model_id}</Badge>
  return <Badge tone="warn">API up · models not trained</Badge>
}

export default function Layout() {
  return (
    <div className="flex min-h-screen">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-50 focus:rounded focus:bg-accent focus:px-3 focus:py-1.5 focus:text-slate-950"
      >
        Skip to content
      </a>
      <aside className="hidden w-64 shrink-0 flex-col border-r border-line bg-panel/60 p-4 md:flex">
        <div className="mb-6 flex items-center gap-2.5 px-1">
          <span className="grid h-9 w-9 place-items-center rounded-lg bg-accent/15 text-accent">
            <Bot className="h-5 w-5" aria-hidden />
          </span>
          <div>
            <p className="text-sm font-bold leading-tight">Text Classification</p>
            <p className="text-xs text-muted">Studio · Traditional ML</p>
          </div>
        </div>
        <nav aria-label="Primary" className="flex flex-1 flex-col gap-1">
          {nav.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              className={({ isActive }) =>
                [
                  'flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm transition-colors',
                  isActive
                    ? 'bg-accent/15 font-medium text-accent'
                    : 'text-muted hover:bg-panel2 hover:text-ink',
                ].join(' ')
              }
            >
              <Icon className="h-4 w-4" aria-hidden />
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="mt-4 border-t border-line pt-3">
          <ApiStatusPill />
          <p className="mt-2 text-[11px] leading-relaxed text-muted">
            Predictions come from fitted scikit-learn pipelines (TF-IDF + linear
            models). No neural networks.
          </p>
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center gap-2 overflow-x-auto border-b border-line bg-panel/80 px-4 py-2 backdrop-blur md:hidden">
          {nav.map(({ to, label, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              className={({ isActive }) =>
                [
                  'whitespace-nowrap rounded px-2.5 py-1 text-xs',
                  isActive ? 'bg-accent/15 text-accent font-medium' : 'text-muted',
                ].join(' ')
              }
            >
              {label}
            </NavLink>
          ))}
        </header>
        <main id="main" className="mx-auto w-full max-w-6xl flex-1 p-4 md:p-8">
          <Outlet />
        </main>
        <footer className="border-t border-line px-6 py-3 text-xs text-muted">
          AI-Powered Text Classification Studio · internship project · UCI SMS Spam
          Collection · scikit-learn
        </footer>
      </div>
    </div>
  )
}
