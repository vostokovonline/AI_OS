/**
 * Filter Toolbar - Controls for filtering graph nodes
 */

import { useGraphStore } from '../../store/graphStore';

export function FilterToolbar() {
  const { filters, setFilters } = useGraphStore();

  return (
    <div className="bg-gray-800 border border-gray-700 text-white px-4 py-3 rounded">
      <div className="text-xs font-bold mb-3 text-gray-400 uppercase">Filters</div>

      <div className="space-y-2 text-xs">
        {/* Status Filters */}
        <div className="space-y-1">
          <div className="text-gray-400 font-medium mb-1">Status</div>
          <label className="flex items-center gap-2 cursor-pointer hover:bg-gray-700 px-2 py-1 rounded">
            <input
              type="checkbox"
              checked={filters.showPending}
              onChange={(e) => setFilters({ showPending: e.target.checked })}
              className="rounded"
            />
            <span className="text-gray-300">Pending</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer hover:bg-gray-700 px-2 py-1 rounded">
            <input
              type="checkbox"
              checked={filters.showActive}
              onChange={(e) => setFilters({ showActive: e.target.checked })}
              className="rounded"
            />
            <span className="text-gray-300">Active</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer hover:bg-gray-700 px-2 py-1 rounded">
            <input
              type="checkbox"
              checked={filters.showDone}
              onChange={(e) => setFilters({ showDone: e.target.checked })}
              className="rounded"
            />
            <span className="text-gray-300">Done</span>
          </label>
        </div>

        {/* Special Filters */}
        <div className="space-y-1 pt-2 border-t border-gray-700">
          <div className="text-gray-400 font-medium mb-1">View</div>

          {/* Atomic Goals Filter */}
          <label className="flex items-center gap-2 cursor-pointer hover:bg-gray-700 px-2 py-1 rounded bg-blue-900/30 border border-blue-500/30">
            <input
              type="checkbox"
              checked={filters.showOnlyAtomic}
              onChange={(e) => setFilters({ showOnlyAtomic: e.target.checked })}
              className="rounded"
            />
            <span className="text-blue-300 font-medium">⚡ Atomic Goals Only</span>
          </label>

          {/* Root Goals Filter */}
          <label className="flex items-center gap-2 cursor-pointer hover:bg-gray-700 px-2 py-1 rounded">
            <input
              type="checkbox"
              checked={filters.showOnlyRoots}
              onChange={(e) => setFilters({ showOnlyRoots: e.target.checked })}
              className="rounded"
            />
            <span className="text-gray-300">Root Goals Only</span>
          </label>
        </div>

        {/* Search */}
        <div className="pt-2 border-t border-gray-700">
          <input
            type="text"
            placeholder="Search goals..."
            value={filters.searchQuery}
            onChange={(e) => setFilters({ searchQuery: e.target.value })}
            className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-white text-xs placeholder-gray-400 focus:outline-none focus:border-blue-500"
          />
        </div>
      </div>
    </div>
  );
}
