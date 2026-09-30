import React from 'react';
import { LayoutDashboard, Sliders, History, Database, HelpCircle, FileText } from 'lucide-react';

export const Sidebar = ({ activeTab, setActiveTab }) => {
  const navItems = [
    { id: 'registration', label: 'Image Registration', icon: LayoutDashboard },
    { id: 'settings', label: 'Algorithm Settings', icon: Sliders },
    { id: 'logs', label: 'Execution Logs', icon: History },
    { id: 'datasets', label: 'Calibrated Datasets', icon: Database },
    { id: 'api', label: 'API Reference', icon: FileText },
    { id: 'help', label: 'Help & Docs', icon: HelpCircle },
  ];

  return (
    <aside className="w-64 border-r border-slate-800 bg-slate-900/50 flex flex-col justify-between p-4 shrink-0 md:flex">
      <div className="space-y-1">
        <div className="px-3 py-2 text-[10px] font-semibold text-slate-500 uppercase tracking-wider">
          Modules
        </div>
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`w-full flex items-center space-x-3 px-3 py-2.5 rounded-lg text-xs font-medium transition-colors ${
                isActive
                  ? 'bg-cyan-500/10 text-cyan-300 border border-cyan-500/20'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
              }`}
            >
              <Icon className={`w-4 h-4 ${isActive ? 'text-cyan-400' : 'text-slate-500'}`} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </div>
    </aside>
  );
};

export default Sidebar;