document.addEventListener('DOMContentLoaded', function() {
    let priceChart = null;
    const predictBtn = document.getElementById('predictBtn');
    const loadingDiv = document.getElementById('loading');
    const resultsDiv = document.getElementById('predictionResults');
    const commoditySelect = document.getElementById('commodity');
    const marketSelect = document.getElementById('market');

    // Populate dropdowns (you'll add your actual options here)
    // Example:
    // commoditySelect.innerHTML = '<option value="Tomato">Tomato</option>...';
    // marketSelect.innerHTML = '<option value="Mumbai">Mumbai</option>...';

    predictBtn.addEventListener('click', async function() {
        const commodity = commoditySelect.value;
        const market = marketSelect.value;
        
        if (!commodity || !market) {
            alert('Please select both commodity and market');
            return;
        }
        
        loadingDiv.style.display = 'block';
        resultsDiv.style.display = 'none';
        
        try {
            const response = await fetch('http://localhost:5000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    commodity: commodity,
                    market: market
                })
            });
            
            if (!response.ok) {
                throw new Error(await response.text());
            }
            
            const predictions = await response.json();
            displayPredictions(commodity, market, predictions);
            
        } catch (error) {
            console.error('Error:', error);
            alert('Error generating predictions: ' + error.message);
        } finally {
            loadingDiv.style.display = 'none';
        }
    });
    
    function displayPredictions(commodity, market, predictions) {
        const predCommodity = document.getElementById('predCommodity');
        const predMarket = document.getElementById('predMarket');
        const predictionDays = document.getElementById('predictionDays');
        
        predCommodity.textContent = commodity;
        predMarket.textContent = market;
        predictionDays.innerHTML = '';
        
        predictions.forEach(pred => {
            const dayElement = document.createElement('div');
            dayElement.className = 'prediction-card';
            dayElement.innerHTML = `
                <div class="prediction-date">${pred.date}</div>
                <div class="prediction-price">₹${pred.price.toFixed(2)}</div>
            `;
            predictionDays.appendChild(dayElement);
        });
        
        resultsDiv.style.display = 'block';
        createChart(predictions);
        resultsDiv.scrollIntoView({ behavior: 'smooth' });
    }
    
    function createChart(predictions) {
        const ctx = document.getElementById('priceChart').getContext('2d');
        
        if (priceChart) priceChart.destroy();
        
        const dates = predictions.map(p => p.date);
        const prices = predictions.map(p => p.price);
        
        priceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: dates,
                datasets: [{
                    label: 'Predicted Price (₹)',
                    data: prices,
                    backgroundColor: 'rgba(76, 175, 80, 0.2)',
                    borderColor: 'rgba(76, 175, 80, 1)',
                    borderWidth: 2,
                    tension: 0.4,
                    pointBackgroundColor: 'rgba(76, 175, 80, 1)',
                    pointRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { beginAtZero: false, title: { display: true, text: 'Price (₹)' } },
                    x: { title: { display: true, text: 'Date' } }
                },
                plugins: {
                    legend: { position: 'top' },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return 'Price: ₹' + context.parsed.y.toFixed(2);
                            }
                        }
                    }
                }
            }
        });
    }
});