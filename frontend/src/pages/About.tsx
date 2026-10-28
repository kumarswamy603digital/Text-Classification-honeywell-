/** About & Documentation: problem, algorithms, architecture, limitations. */
import { Link } from 'react-router-dom'
import { Card, Badge } from '../components/ui/primitives'

export default function About() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">About & Documentation</h1>
        <p className="mt-1 text-sm text-muted">
          AI-Powered Text Classification Studio — an end-to-end traditional ML project.
        </p>
      </div>

      <Card title="Problem statement">
        <p className="text-sm leading-relaxed text-muted">
          Spam filtering is a classic text-classification problem: given a short message,
          decide whether it is <Badge tone="danger">spam</Badge> or{' '}
          <Badge tone="success">legitimate (ham)</Badge>. This project implements the full
          ML lifecycle — dataset acquisition and validation, EDA, preprocessing, TF-IDF
          feature extraction, training four traditional scikit-learn classifiers,
          hyperparameter tuning with cross-validation, honest evaluation on a held-out
          test split, and serving real fitted pipelines through a FastAPI + React
          application. No neural networks, transformers or hosted LLMs are used.
        </p>
      </Card>

      <Card title="The algorithms">
        <div className="grid gap-3 text-sm md:grid-cols-2">
          {[
            {
              n: 'TF-IDF (features)',
              d: 'Each message becomes a sparse vector: term frequency scaled by inverse document frequency, so rare-but-distinctive words count more than generic ones. Unigrams + bigrams, fitted only on training data.',
            },
            {
              n: 'Multinomial Naive Bayes',
              d: 'Probabilistic classifier applying Bayes\' theorem with a feature-independence assumption. Extremely fast; strong text baseline. Smoothing (α) tuned by CV.',
            },
            {
              n: 'Logistic Regression',
              d: 'Linear model learning a weight per TF-IDF feature; the sigmoid of the weighted sum is a calibrated class probability. Regularisation strength C tuned by CV.',
            },
            {
              n: 'Linear SVM',
              d: 'Finds the maximum-margin separating hyperplane. Excellent on sparse text. Its decision scores are NOT probabilities, so the app deliberately shows no confidence for this model.',
            },
            {
              n: 'Complement Naive Bayes',
              d: 'A variant of NB designed for imbalanced text: it uses statistics of the *complement* of each class, which stabilises estimates when one class dominates.',
            },
            {
              n: 'Macro-F1 & confusion matrix',
              d: 'Macro-F1 averages per-class F1 so the minority class (spam) weighs as much as the majority. The confusion matrix separates false positives from false negatives — the two very different kinds of mistakes.',
            },
          ].map((a) => (
            <div key={a.n} className="rounded-lg border border-line bg-panel2/40 p-3">
              <p className="font-semibold">{a.n}</p>
              <p className="mt-1 text-xs leading-relaxed text-muted">{a.d}</p>
            </div>
          ))}
        </div>
      </Card>

      <Card title="Architecture">
        <pre className="overflow-x-auto rounded-lg bg-panel2/60 p-4 font-mono text-xs leading-relaxed">
{`React + Vite + TypeScript + Tailwind (frontend, port 5173)
        │  typed fetch client (JSON over HTTPS/CORS)
        ▼
FastAPI + Pydantic (backend, port 8000)
  ├── /api/v1/classify · /classify/batch ──► PredictionService
  ├── /api/v1/models · /models/selected          │
  ├── /api/v1/analytics/*  ◄── AnalyticsService  │
  └── /api/v1/history      ◄── SQLite (optional) │
                                                 ▼
                             ModelRegistry ──► joblib pipelines
                             [TextPreprocessor → TfidfVectorizer → Classifier]
                                                 ▲
scripts: download_data → explore_data → train_models → evaluate_models
        (UCI SMS Spam Collection)   (GridSearchCV, macro-F1 selection)`}
        </pre>
        <p className="mt-3 text-xs text-muted">
          The same fitted pipeline object that produced the saved metrics performs every
          live prediction — preprocessing, featurization and the classifier are one
          artifact, so training and inference can never drift apart.
        </p>
      </Card>

      <Card title="Reproducing the experiments">
        <ol className="list-decimal space-y-1 pl-5 text-sm text-muted">
          <li>
            Backend: <code className="font-mono text-accent">pip install -r requirements.txt</code>
          </li>
          <li>
            <code className="font-mono text-accent">python scripts/download_data.py</code> — acquire + validate the dataset
          </li>
          <li>
            <code className="font-mono text-accent">python scripts/explore_data.py</code> — EDA artifacts &amp; figures
          </li>
          <li>
            <code className="font-mono text-accent">python scripts/train_models.py</code> — tune, select, evaluate, save
          </li>
          <li>
            <code className="font-mono text-accent">uvicorn app.main:app --port 8000</code> +{' '}
            <code className="font-mono text-accent">npm run dev</code> in frontend/
          </li>
        </ol>
        <p className="mt-3 text-sm text-muted">
          Full documentation lives in <code className="font-mono">docs/</code>:{' '}
          <Link to="/" className="text-accent hover:underline">
            architecture · dataset card · model report · API reference · deployment · demo script
          </Link>
          . Interactive API docs:{' '}
          <a href="http://localhost:8000/docs" target="_blank" rel="noreferrer" className="text-accent hover:underline">
            http://localhost:8000/docs
          </a>
          .
        </p>
      </Card>

      <Card title="Limitations & future work">
        <ul className="list-disc space-y-1.5 pl-5 text-sm text-muted">
          <li>Bag-of-words TF-IDF ignores word order beyond bigrams and any semantics a paraphrase might carry.</li>
          <li>Trained on 2007-era UK SMS spam; modern phishing styles (URL shorteners, messaging-app lures) will drift.</li>
          <li>Linear SVM reports no probability; adding CalibratedClassifierCV would be a measured next step.</li>
          <li>Binary task only — the registry design supports multiclass but the dataset is binary.</li>
          <li>Prediction history is a local SQLite file, not suitable for multi-instance deployments.</li>
        </ul>
      </Card>
    </div>
  )
}
