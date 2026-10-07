import { Routes, Route, Navigate } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import Dashboard from './pages/Dashboard'
import DataLoader from './pages/DataLoader'
import EDA from './pages/EDA'
import FeatureEngineering from './pages/FeatureEngineering'
import LinearRegression from './pages/LinearRegression'
import LogisticRegression from './pages/LogisticRegression'
import Regularization from './pages/Regularization'
import DecisionTree from './pages/DecisionTree'
import Ensemble from './pages/Ensemble'

export default function App() {
  return (
    <div className="layout">
      <Sidebar />
      <div className="content">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/data-loading" element={<DataLoader />} />
          <Route path="/eda" element={<EDA />} />
          <Route path="/feature-engineering" element={<FeatureEngineering />} />
          <Route path="/linear-regression" element={<LinearRegression />} />
          <Route path="/logistic-regression" element={<LogisticRegression />} />
          <Route path="/regularization" element={<Regularization />} />
          <Route path="/decision-tree" element={<DecisionTree />} />
          <Route path="/ensemble" element={<Ensemble />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </div>
    </div>
  )
}
