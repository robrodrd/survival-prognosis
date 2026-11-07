if (!window.__tableScrollSyncInitialized) {
    window.__tableScrollSyncInitialized = true;

    function initScrollSync() {
        // Automatically find the table on the page
        const tables = document.querySelectorAll('table');
        tables.forEach(table => {
            // Walk up the DOM tree to find the actual scrolling container
            let container = table.parentElement;
            while (container && container !== document.body) {
                const style = window.getComputedStyle(container);
                if (style.overflowY === 'auto' || style.overflowY === 'scroll' || container.id === 'inspector-table-container') {
                    
                    // Attach listener only once per container instance
                    if (!container.__scrollSyncAttached) {
                        container.__scrollSyncAttached = true;
                        
                        container.addEventListener('scroll', function() {
                            const tbody = container.querySelector('tbody');
                            if (!tbody) return;
                            const rows = tbody.querySelectorAll('tr');
                            if (rows.length === 0) return;

                            const containerRect = container.getBoundingClientRect();
                            let closestRowIndex = 0;
                            let minDistance = Infinity;

                            rows.forEach((row, idx) => {
                                const rect = row.getBoundingClientRect();
                                const distance = Math.abs(rect.top - containerRect.top);
                                if (distance < minDistance) {
                                    minDistance = distance;
                                    closestRowIndex = idx;
                                }
                            });

                            const baseOffset = parseInt(container.getAttribute('data-viewport-offset') || '0', 10);
                            const currentRow = baseOffset + closestRowIndex + 1;

                            const input = document.getElementById('row-number-input');
                            if (input && document.activeElement !== input) {
                                input.value = currentRow;
                            }
                        });
                    }
                    break;
                }
                container = container.parentElement;
            }
        });
    }

    // Watch for DOM changes (so it survives Reflex re-renders and tab switches)
    const observer = new MutationObserver(() => {
        initScrollSync();
    });
    
    observer.observe(document.body, { childList: true, subtree: true });

    // Initial run after page load
    window.addEventListener('DOMContentLoaded', initScrollSync);
    setTimeout(initScrollSync, 500);
}