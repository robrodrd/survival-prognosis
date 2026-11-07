setInterval(() => {
    const container = document.getElementById('inspector-table-container');
    
    if (container && !container._hasScrollListener) {
        container._hasScrollListener = true;
        console.log("Scroll sync script successfully loaded and active!");
        
        container.addEventListener('scroll', function() {
            let input = document.querySelector('#row-number-input input');
            if (!input) {
                input = document.getElementById('row-number-input');
            }
            
            const totalRowsEl = document.getElementById('total-rows-count');
            const totalRows = totalRowsEl ? parseInt(totalRowsEl.innerText) || 1 : 1;

            if (container && input) {
                const scrollTop = container.scrollTop;
                const scrollHeight = container.scrollHeight - container.clientHeight;
                
                if (scrollHeight <= 0) return;
                
                const scrollFraction = scrollTop / scrollHeight;
                const calculatedRow = Math.min(totalRows, Math.max(1, Math.round(scrollFraction * (totalRows - 1)) + 1));
                
                if (document.activeElement !== input) {
                    input.value = calculatedRow;
                    input.dispatchEvent(new Event('input', { bubbles: true }));
                }
            }
        });
    }
}, 500);