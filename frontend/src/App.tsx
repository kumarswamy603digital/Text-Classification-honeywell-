import { Navigate, Route, Routes } from 'react-router-dom'
import Layout from './components/layout/Layout'
import Dashboard from './pages/Dashboard'
import Studio from './pages/Studio'
import Batch from './pages/Batch'
import Comparison from './pages/Comparison'
import DatasetExplorer from './pages/DatasetExplorer'
import About from './pages/About'

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="studio" element={<Studio />} />
        <Route path="batch" element={<Batch />} />
        <Route path="compare" element={<Comparison />} />
        <Route path="dataset" element={<DatasetExplorer />} />
        <Route path="about" element={<About />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  )
}
