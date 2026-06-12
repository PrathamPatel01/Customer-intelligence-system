import React, { useState } from "react";
import { 
  Activity, 
  AlertTriangle, 
  ArrowRight, 
  BarChart3, 
  ChevronDown, 
  DollarSign, 
  ShieldAlert, 
  User, 
  Zap,
  CheckCircle2,
  TrendingUp,
  TrendingDown
} from "lucide-react";

export function DarkPro() {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showResults, setShowResults] = useState(true);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setTimeout(() => {
      setIsSubmitting(false);
      setShowResults(true);
    }, 1200);
  };

  return (
    <div className="dark min-h-screen bg-[#0a0c10] text-slate-300 font-sans selection:bg-blue-500/30">
      <div className="max-w-6xl mx-auto p-4 md:p-8">
        
        {/* Header */}
        <header className="flex items-center justify-between mb-8 pb-6 border-b border-white/5">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded bg-gradient-to-br from-indigo-500 to-blue-600 flex items-center justify-center shadow-[0_0_20px_rgba(79,70,229,0.2)]">
              <Activity className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-semibold text-white tracking-tight">Customer Intelligence</h1>
              <p className="text-xs text-slate-500 uppercase tracking-wider font-medium mt-0.5">Predictive Analysis System</p>
            </div>
          </div>
          <div className="flex items-center gap-4 text-sm">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
              <span className="text-slate-400">Model Active v2.4</span>
            </div>
            <div className="h-4 w-px bg-white/10"></div>
            <span className="text-slate-400 font-mono text-xs">SYS_ID: 9481-A</span>
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* Input Form Column */}
          <div className="lg:col-span-5 space-y-6">
            <div className="bg-[#11141a] rounded-xl border border-white/5 p-6 shadow-xl relative overflow-hidden">
              <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-blue-500/50 to-transparent"></div>
              
              <div className="flex items-center gap-2 mb-6">
                <User className="w-4 h-4 text-blue-400" />
                <h2 className="text-sm font-semibold text-white uppercase tracking-wider">Target Profile</h2>
              </div>

              <form onSubmit={handleSubmit} className="space-y-5">
                {/* Numeric Inputs */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1.5">
                    <label className="text-xs text-slate-400 font-medium ml-1">Tenure (Months)</label>
                    <div className="relative">
                      <input 
                        type="number" 
                        defaultValue={24}
                        className="w-full bg-[#0a0c10] border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500/50 transition-all font-mono"
                      />
                    </div>
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-xs text-slate-400 font-medium ml-1">Monthly Charges ($)</label>
                    <div className="relative">
                      <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none">
                        <DollarSign className="w-3 h-3 text-slate-500" />
                      </div>
                      <input 
                        type="number" 
                        defaultValue={89.50}
                        step="0.01"
                        className="w-full bg-[#0a0c10] border border-white/10 rounded-lg pl-7 pr-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500/50 transition-all font-mono"
                      />
                    </div>
                  </div>
                </div>

                {/* Categorical Inputs */}
                <div className="space-y-4 pt-2">
                  <div className="space-y-1.5">
                    <label className="text-xs text-slate-400 font-medium ml-1">Contract Type</label>
                    <div className="relative">
                      <select className="w-full appearance-none bg-[#0a0c10] border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500/50 transition-all">
                        <option>Month-to-month</option>
                        <option>One year</option>
                        <option>Two year</option>
                      </select>
                      <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500 pointer-events-none" />
                    </div>
                  </div>

                  <div className="space-y-1.5">
                    <label className="text-xs text-slate-400 font-medium ml-1">Internet Service</label>
                    <div className="relative">
                      <select defaultValue="Fiber optic" className="w-full appearance-none bg-[#0a0c10] border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500/50 transition-all">
                        <option>DSL</option>
                        <option>Fiber optic</option>
                        <option>No</option>
                      </select>
                      <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500 pointer-events-none" />
                    </div>
                  </div>

                  <div className="space-y-1.5">
                    <label className="text-xs text-slate-400 font-medium ml-1">Payment Method</label>
                    <div className="relative">
                      <select defaultValue="Electronic check" className="w-full appearance-none bg-[#0a0c10] border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500/50 transition-all">
                        <option>Electronic check</option>
                        <option>Mailed check</option>
                        <option>Bank transfer (automatic)</option>
                        <option>Credit card (automatic)</option>
                      </select>
                      <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500 pointer-events-none" />
                    </div>
                  </div>
                </div>

                {/* Toggles */}
                <div className="grid grid-cols-2 gap-4 pt-2">
                  <div className="space-y-2">
                    <label className="text-xs text-slate-400 font-medium ml-1">Paperless Billing</label>
                    <div className="flex bg-[#0a0c10] border border-white/10 rounded-lg p-1">
                      <button type="button" className="flex-1 py-1 text-xs font-medium rounded text-white bg-white/10 shadow-sm">Yes</button>
                      <button type="button" className="flex-1 py-1 text-xs font-medium rounded text-slate-500 hover:text-slate-300 transition-colors">No</button>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <label className="text-xs text-slate-400 font-medium ml-1">Senior Citizen</label>
                    <div className="flex bg-[#0a0c10] border border-white/10 rounded-lg p-1">
                      <button type="button" className="flex-1 py-1 text-xs font-medium rounded text-slate-500 hover:text-slate-300 transition-colors">Yes</button>
                      <button type="button" className="flex-1 py-1 text-xs font-medium rounded text-white bg-white/10 shadow-sm">No</button>
                    </div>
                  </div>
                  <div className="space-y-2 col-span-2">
                    <label className="text-xs text-slate-400 font-medium ml-1">Has Partner</label>
                    <div className="flex bg-[#0a0c10] border border-white/10 rounded-lg p-1">
                      <button type="button" className="flex-1 py-1 text-xs font-medium rounded text-white bg-white/10 shadow-sm">Yes</button>
                      <button type="button" className="flex-1 py-1 text-xs font-medium rounded text-slate-500 hover:text-slate-300 transition-colors">No</button>
                    </div>
                  </div>
                </div>

                <div className="pt-4">
                  <button 
                    type="submit" 
                    disabled={isSubmitting}
                    className="w-full bg-blue-600 hover:bg-blue-500 text-white font-medium py-2.5 rounded-lg transition-all flex items-center justify-center gap-2 group relative overflow-hidden"
                  >
                    <div className="absolute inset-0 bg-gradient-to-r from-blue-600 to-indigo-600 opacity-0 group-hover:opacity-100 transition-opacity"></div>
                    <span className="relative flex items-center gap-2">
                      {isSubmitting ? (
                        <>
                          <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                          Processing Data...
                        </>
                      ) : (
                        <>
                          <Zap className="w-4 h-4" />
                          Run Prediction Model
                        </>
                      )}
                    </span>
                  </button>
                </div>
              </form>
            </div>
          </div>

          {/* Results Column */}
          <div className="lg:col-span-7">
            {showResults ? (
              <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-700 fill-mode-both">
                
                {/* Main Metrics Grid */}
                <div className="grid grid-cols-2 gap-4">
                  
                  {/* Churn Risk Card */}
                  <div className="bg-[#11141a] rounded-xl border border-rose-500/20 p-5 relative overflow-hidden group">
                    <div className="absolute -right-6 -top-6 w-24 h-24 bg-rose-500/10 rounded-full blur-xl group-hover:bg-rose-500/20 transition-all"></div>
                    <div className="flex justify-between items-start mb-4 relative z-10">
                      <div className="flex items-center gap-2">
                        <AlertTriangle className="w-4 h-4 text-rose-500" />
                        <h3 className="text-sm font-medium text-slate-300">Churn Risk</h3>
                      </div>
                      <span className="text-xs font-mono bg-rose-500/10 text-rose-400 px-2 py-0.5 rounded border border-rose-500/20">HIGH</span>
                    </div>
                    <div className="relative z-10 flex items-baseline gap-2">
                      <span className="text-5xl font-light text-white font-mono tracking-tight">78<span className="text-2xl text-rose-500">%</span></span>
                      <TrendingUp className="w-4 h-4 text-rose-500" />
                    </div>
                    <div className="mt-4 pt-4 border-t border-white/5 relative z-10">
                      <div className="w-full h-1.5 bg-[#0a0c10] rounded-full overflow-hidden">
                        <div className="h-full bg-gradient-to-r from-rose-600 to-amber-500 w-[78%]"></div>
                      </div>
                      <p className="text-xs text-slate-500 mt-2">Probability of leaving within 30 days</p>
                    </div>
                  </div>

                  {/* Predicted Value Card */}
                  <div className="bg-[#11141a] rounded-xl border border-emerald-500/20 p-5 relative overflow-hidden group">
                    <div className="absolute -right-6 -top-6 w-24 h-24 bg-emerald-500/10 rounded-full blur-xl group-hover:bg-emerald-500/20 transition-all"></div>
                    <div className="flex justify-between items-start mb-4 relative z-10">
                      <div className="flex items-center gap-2">
                        <BarChart3 className="w-4 h-4 text-emerald-500" />
                        <h3 className="text-sm font-medium text-slate-300">Predicted LTV</h3>
                      </div>
                      <span className="text-xs font-mono bg-emerald-500/10 text-emerald-400 px-2 py-0.5 rounded border border-emerald-500/20">TIER 1</span>
                    </div>
                    <div className="relative z-10 flex items-baseline gap-2">
                      <span className="text-5xl font-light text-white font-mono tracking-tight">$4.2<span className="text-2xl text-emerald-500">k</span></span>
                      <TrendingDown className="w-4 h-4 text-rose-500 opacity-50" />
                    </div>
                    <div className="mt-4 pt-4 border-t border-white/5 relative z-10">
                      <div className="w-full h-1.5 bg-[#0a0c10] rounded-full overflow-hidden">
                        <div className="h-full bg-gradient-to-r from-emerald-600 to-teal-400 w-[85%]"></div>
                      </div>
                      <p className="text-xs text-slate-500 mt-2">Estimated 24-month revenue potential</p>
                    </div>
                  </div>
                </div>

                {/* Segment & Recommendation */}
                <div className="bg-[#11141a] rounded-xl border border-white/5 p-6 shadow-xl">
                  <h3 className="text-sm font-semibold text-white uppercase tracking-wider mb-6 flex items-center gap-2">
                    <ShieldAlert className="w-4 h-4 text-amber-500" />
                    Strategic Assessment
                  </h3>
                  
                  <div className="bg-[#0a0c10] rounded-lg border border-white/5 p-5 mb-6">
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                      <div>
                        <p className="text-xs text-slate-500 mb-1">Business Segment</p>
                        <div className="flex items-center gap-2">
                          <h4 className="text-lg font-medium text-white">High Value – At Risk</h4>
                          <span className="flex h-2 w-2 relative">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75"></span>
                            <span className="relative inline-flex rounded-full h-2 w-2 bg-amber-500"></span>
                          </span>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="text-xs text-slate-500 mb-1">Intervention Priority</p>
                        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded bg-amber-500/10 text-amber-400 text-xs font-mono border border-amber-500/20">
                          CRITICAL
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="space-y-4">
                    <p className="text-sm text-slate-300 leading-relaxed">
                      Customer shows strong indicators of churn despite high historical value. 
                      Month-to-month contract coupled with Fiber Optic service and high monthly 
                      charges ($89.50) is a known flight-risk pattern.
                    </p>
                    
                    <div className="pt-4 border-t border-white/5">
                      <h4 className="text-xs font-semibold text-white uppercase tracking-wider mb-3">Recommended Actions</h4>
                      <ul className="space-y-3">
                        <li className="flex items-start gap-3 bg-blue-500/5 border border-blue-500/10 rounded-lg p-3">
                          <CheckCircle2 className="w-4 h-4 text-blue-400 shrink-0 mt-0.5" />
                          <div>
                            <span className="text-sm font-medium text-white block mb-0.5">Offer 1-Year Contract Upgrade</span>
                            <span className="text-xs text-slate-400">Provide a 15% discount for locking in a 1-year contract to secure LTV.</span>
                          </div>
                        </li>
                        <li className="flex items-start gap-3 bg-white/5 border border-white/5 rounded-lg p-3">
                          <CheckCircle2 className="w-4 h-4 text-slate-500 shrink-0 mt-0.5" />
                          <div>
                            <span className="text-sm font-medium text-slate-300 block mb-0.5">Proactive Service Check</span>
                            <span className="text-xs text-slate-500">Schedule an automated technical review to ensure Fiber Optic stability.</span>
                          </div>
                        </li>
                      </ul>
                    </div>
                  </div>
                  
                  <div className="mt-6 flex gap-3">
                    <button className="flex-1 bg-white/5 hover:bg-white/10 text-white border border-white/10 text-sm font-medium py-2 rounded-lg transition-colors">
                      Export Report
                    </button>
                    <button className="flex-1 bg-blue-600/20 hover:bg-blue-600/30 text-blue-400 border border-blue-500/30 text-sm font-medium py-2 rounded-lg transition-colors flex items-center justify-center gap-2">
                      Apply Campaign
                      <ArrowRight className="w-4 h-4" />
                    </button>
                  </div>
                </div>

              </div>
            ) : (
              <div className="h-full flex flex-col items-center justify-center border-2 border-dashed border-white/5 rounded-xl bg-[#11141a]/50 text-center p-12 min-h-[500px]">
                <Activity className="w-12 h-12 text-slate-600 mb-4" />
                <h3 className="text-lg font-medium text-slate-300 mb-2">Awaiting Parameters</h3>
                <p className="text-sm text-slate-500 max-w-sm">
                  Adjust the customer profile attributes on the left and run the prediction model to generate insights.
                </p>
              </div>
            )}
          </div>
          
        </div>
      </div>
    </div>
  );
}
