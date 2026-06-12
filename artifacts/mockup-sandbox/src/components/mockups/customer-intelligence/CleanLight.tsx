import React, { useState } from 'react';
import { 
  Activity, 
  AlertCircle, 
  ArrowRight, 
  BarChart3, 
  CheckCircle2, 
  ChevronDown,
  CreditCard, 
  FileText, 
  Globe, 
  HelpCircle, 
  Search,
  ShieldAlert,
  Sparkles,
  TrendingUp,
  User,
  Zap,
  Building,
  Mail,
  UserCheck
} from 'lucide-react';

export function CleanLight() {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showResults, setShowResults] = useState(true);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setTimeout(() => {
      setIsSubmitting(false);
      setShowResults(true);
    }, 800);
  };

  return (
    <div className="min-h-screen bg-[#FAFAFA] text-slate-900 font-sans p-6 md:p-10 selection:bg-blue-100 selection:text-blue-900">
      <div className="max-w-5xl mx-auto space-y-8">
        
        {/* Header */}
        <header className="flex flex-col md:flex-row md:items-end justify-between gap-4 pb-6 border-b border-slate-200">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <div className="w-8 h-8 rounded-md bg-blue-600 flex items-center justify-center text-white shadow-sm">
                <Activity size={18} strokeWidth={2.5} />
              </div>
              <h1 className="text-xl font-semibold tracking-tight">Customer Intelligence</h1>
            </div>
            <p className="text-sm text-slate-500">Assess churn risk and predicted lifetime value for individual accounts.</p>
          </div>
          <div className="flex items-center gap-3">
            <button className="px-4 py-2 text-sm font-medium text-slate-600 bg-white border border-slate-200 rounded-md hover:bg-slate-50 transition-colors shadow-sm">
              Batch Import
            </button>
            <button className="px-4 py-2 text-sm font-medium text-white bg-slate-900 rounded-md hover:bg-slate-800 transition-colors shadow-sm">
              View Analytics
            </button>
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* Left Column: Form */}
          <div className="lg:col-span-7 space-y-6">
            <form onSubmit={handleSubmit} className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm space-y-8">
              <div>
                <h2 className="text-base font-semibold mb-4 flex items-center gap-2">
                  <User size={16} className="text-slate-400" />
                  Account Details
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                  <div className="space-y-1.5">
                    <label className="text-sm font-medium text-slate-700">Tenure (Months)</label>
                    <input 
                      type="number" 
                      defaultValue={14}
                      className="w-full px-3 py-2 text-sm bg-slate-50 border border-slate-200 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all"
                    />
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-sm font-medium text-slate-700">Monthly Charges ($)</label>
                    <input 
                      type="number" 
                      defaultValue={85.50}
                      className="w-full px-3 py-2 text-sm bg-slate-50 border border-slate-200 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all"
                    />
                  </div>
                </div>
              </div>

              <div className="h-px bg-slate-100" />

              <div>
                <h2 className="text-base font-semibold mb-4 flex items-center gap-2">
                  <FileText size={16} className="text-slate-400" />
                  Service & Billing
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                  <div className="space-y-1.5">
                    <label className="text-sm font-medium text-slate-700">Contract Type</label>
                    <div className="relative">
                      <select className="w-full px-3 py-2 text-sm bg-slate-50 border border-slate-200 rounded-md appearance-none focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all">
                        <option>Month-to-month</option>
                        <option>One year</option>
                        <option>Two year</option>
                      </select>
                      <ChevronDown size={14} className="absolute right-3 top-3 text-slate-400 pointer-events-none" />
                    </div>
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-sm font-medium text-slate-700">Internet Service</label>
                    <div className="relative">
                      <select className="w-full px-3 py-2 text-sm bg-slate-50 border border-slate-200 rounded-md appearance-none focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all">
                        <option>Fiber optic</option>
                        <option>DSL</option>
                        <option>No</option>
                      </select>
                      <ChevronDown size={14} className="absolute right-3 top-3 text-slate-400 pointer-events-none" />
                    </div>
                  </div>
                  <div className="space-y-1.5 md:col-span-2">
                    <label className="text-sm font-medium text-slate-700">Payment Method</label>
                    <div className="relative">
                      <select className="w-full px-3 py-2 text-sm bg-slate-50 border border-slate-200 rounded-md appearance-none focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all">
                        <option>Electronic check</option>
                        <option>Mailed check</option>
                        <option>Bank transfer (automatic)</option>
                        <option>Credit card (automatic)</option>
                      </select>
                      <ChevronDown size={14} className="absolute right-3 top-3 text-slate-400 pointer-events-none" />
                    </div>
                  </div>
                </div>
              </div>

              <div className="h-px bg-slate-100" />

              <div>
                <h2 className="text-base font-semibold mb-4 flex items-center gap-2">
                  <UserCheck size={16} className="text-slate-400" />
                  Demographics & Flags
                </h2>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                  <label className="flex items-start gap-3 p-3 border border-slate-200 rounded-md cursor-pointer hover:bg-slate-50 transition-colors">
                    <div className="pt-0.5">
                      <input type="checkbox" defaultChecked className="w-4 h-4 text-blue-600 rounded border-slate-300 focus:ring-blue-500" />
                    </div>
                    <div>
                      <div className="text-sm font-medium text-slate-700">Paperless Billing</div>
                    </div>
                  </label>
                  <label className="flex items-start gap-3 p-3 border border-slate-200 rounded-md cursor-pointer hover:bg-slate-50 transition-colors">
                    <div className="pt-0.5">
                      <input type="checkbox" className="w-4 h-4 text-blue-600 rounded border-slate-300 focus:ring-blue-500" />
                    </div>
                    <div>
                      <div className="text-sm font-medium text-slate-700">Senior Citizen</div>
                    </div>
                  </label>
                  <label className="flex items-start gap-3 p-3 border border-slate-200 rounded-md cursor-pointer hover:bg-slate-50 transition-colors">
                    <div className="pt-0.5">
                      <input type="checkbox" className="w-4 h-4 text-blue-600 rounded border-slate-300 focus:ring-blue-500" />
                    </div>
                    <div>
                      <div className="text-sm font-medium text-slate-700">Partner</div>
                    </div>
                  </label>
                </div>
              </div>

              <div className="pt-4 flex items-center justify-end">
                <button 
                  type="submit" 
                  disabled={isSubmitting}
                  className="flex items-center gap-2 px-5 py-2.5 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-all disabled:opacity-70 shadow-sm"
                >
                  {isSubmitting ? (
                    <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  ) : (
                    <Sparkles size={16} />
                  )}
                  Generate Prediction
                </button>
              </div>
            </form>
          </div>

          {/* Right Column: Results */}
          <div className="lg:col-span-5 space-y-6">
            {showResults ? (
              <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
                <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm space-y-6 relative overflow-hidden">
                  {/* Decorative top border */}
                  <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-red-500 via-amber-400 to-blue-500" />
                  
                  <div className="flex items-start justify-between">
                    <div>
                      <h2 className="text-lg font-semibold tracking-tight">Analysis Result</h2>
                      <p className="text-sm text-slate-500 mt-1">Predictions based on model v2.4.1</p>
                    </div>
                    <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-red-50 text-red-700 border border-red-100">
                      <div className="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse" />
                      High Risk
                    </span>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-4 rounded-lg bg-slate-50 border border-slate-100">
                      <div className="text-xs font-medium text-slate-500 mb-1 uppercase tracking-wider flex items-center gap-1.5">
                        <ShieldAlert size={14} className="text-red-500" />
                        Churn Probability
                      </div>
                      <div className="flex items-baseline gap-2">
                        <span className="text-3xl font-bold tracking-tight text-slate-900">78<span className="text-xl text-slate-500">%</span></span>
                      </div>
                      <div className="mt-2 w-full h-1.5 bg-slate-200 rounded-full overflow-hidden">
                        <div className="h-full bg-red-500 rounded-full" style={{ width: '78%' }} />
                      </div>
                    </div>

                    <div className="p-4 rounded-lg bg-slate-50 border border-slate-100">
                      <div className="text-xs font-medium text-slate-500 mb-1 uppercase tracking-wider flex items-center gap-1.5">
                        <TrendingUp size={14} className="text-blue-500" />
                        Predicted Value
                      </div>
                      <div className="flex items-baseline gap-2">
                        <span className="text-3xl font-bold tracking-tight text-slate-900"><span className="text-xl text-slate-500 mr-0.5">$</span>4,250</span>
                      </div>
                      <div className="mt-2 text-xs font-medium text-slate-600 flex items-center gap-1">
                        <span className="text-emerald-600">Top 15%</span> of cohort
                      </div>
                    </div>
                  </div>

                  <div className="pt-4 border-t border-slate-100">
                    <div className="text-sm font-medium text-slate-900 mb-2">Customer Segment</div>
                    <div className="inline-flex items-center px-3 py-1.5 rounded-md text-sm font-medium bg-slate-900 text-white shadow-sm">
                      High Value – At Risk
                    </div>
                  </div>

                  <div className="p-4 rounded-lg bg-amber-50/50 border border-amber-200/60">
                    <div className="flex gap-3">
                      <AlertCircle className="text-amber-600 shrink-0 mt-0.5" size={18} />
                      <div>
                        <h4 className="text-sm font-semibold text-amber-900">Recommended Action</h4>
                        <p className="text-sm text-amber-800 mt-1 leading-relaxed">
                          Immediate retention intervention required. Customer is highly sensitive to monthly charges on month-to-month plan. Offer a <strong>20% discount</strong> on a 1-year contract extension.
                        </p>
                        <button className="mt-3 text-sm font-medium text-amber-700 hover:text-amber-800 flex items-center gap-1 group transition-colors">
                          Apply Retention Playbook
                          <ArrowRight size={14} className="group-hover:translate-x-0.5 transition-transform" />
                        </button>
                      </div>
                    </div>
                  </div>

                </div>
              </div>
            ) : (
              <div className="h-full min-h-[400px] flex items-center justify-center border border-slate-200 border-dashed rounded-xl bg-slate-50/50 text-slate-400">
                <div className="text-center space-y-3">
                  <BarChart3 size={32} className="mx-auto text-slate-300" />
                  <p className="text-sm">Submit the form to generate intelligence.</p>
                </div>
              </div>
            )}
          </div>

        </div>
      </div>
    </div>
  );
}
