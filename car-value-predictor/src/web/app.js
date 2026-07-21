document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const form = document.getElementById('predictor-form');
    const predictBtn = document.getElementById('btn-predict');
    
    // Sliders & Label Displays
    const yearInput = document.getElementById('year');
    const yearVal = document.getElementById('year_val');
    
    const kmsDrivenInput = document.getElementById('kms_driven');
    const kmsDrivenVal = document.getElementById('kms_driven_val');
    
    const engineInput = document.getElementById('engine');
    const engineVal = document.getElementById('engine_val');
    
    const maxPowerInput = document.getElementById('max_power');
    const maxPowerVal = document.getElementById('max_power_val');
    
    const mileageInput = document.getElementById('mileage');
    const mileageVal = document.getElementById('mileage_val');
    
    // Theme & Currency Toggles
    const themeToggle = document.getElementById('theme-toggle');
    const currencyInrBtn = document.getElementById('currency-inr');
    const currencyUsdBtn = document.getElementById('currency-usd');
    
    // Result Elements
    const resultsPlaceholder = document.getElementById('results-placeholder');
    const resultsContent = document.getElementById('results-content');
    const predictedPriceText = document.getElementById('predicted-price-text');
    const carAgeText = document.getElementById('car-age-text');
    const retentionPercentText = document.getElementById('retention-percent-text');
    const progressRingBar = document.getElementById('progress-ring-bar');
    
    // Breakdown Items
    const breakdownOriginal = document.getElementById('breakdown-original');
    const breakdownPredicted = document.getElementById('breakdown-predicted');
    const breakdownDepreciation = document.getElementById('breakdown-depreciation');
    const insightDescriptionText = document.getElementById('insight-description-text');

    // Global state
    let currentCurrency = 'INR'; // 'INR' or 'USD'
    let predictionResult = null; // Stores last API response

    // Constants
    const USD_RATE = 1200; // $1200 USD per 1 Lakh INR

    // --- Formatters ---
    const formatKms = (val) => {
        return parseInt(val).toLocaleString('en-IN') + ' km';
    };

    const formatCurrency = (valLakhs, currencyType) => {
        if (currencyType === 'INR') {
            return `₹${parseFloat(valLakhs).toFixed(2)} Lakhs`;
        } else {
            const usdValue = valLakhs * USD_RATE;
            return `$${Math.round(usdValue).toLocaleString('en-US')}`;
        }
    };

    // --- Slider Event Listeners ---
    yearInput.addEventListener('input', (e) => {
        const year = parseInt(e.target.value);
        const age = 2026 - year;
        yearVal.textContent = `${year} (${age} yr old)`;
    });

    kmsDrivenInput.addEventListener('input', (e) => {
        kmsDrivenVal.textContent = formatKms(e.target.value);
    });

    engineInput.addEventListener('input', (e) => {
        engineVal.textContent = `${e.target.value} cc`;
    });

    maxPowerInput.addEventListener('input', (e) => {
        maxPowerVal.textContent = `${e.target.value} bhp`;
    });

    mileageInput.addEventListener('input', (e) => {
        mileageVal.textContent = `${parseFloat(e.target.value).toFixed(1)} kmpl`;
    });

    // --- Currency Toggle Logic ---
    const updateUIWithCurrency = (currency) => {
        currentCurrency = currency;
        
        // If prediction exists, re-render results panel in the new currency
        if (predictionResult) {
            renderResults();
        }
    };

    currencyInrBtn.addEventListener('click', () => {
        currencyInrBtn.classList.add('active');
        currencyUsdBtn.classList.remove('active');
        updateUIWithCurrency('INR');
    });

    currencyUsdBtn.addEventListener('click', () => {
        currencyUsdBtn.classList.add('active');
        currencyInrBtn.classList.remove('active');
        updateUIWithCurrency('USD');
    });

    // --- Theme Toggle Logic ---
    themeToggle.addEventListener('click', () => {
        const html = document.documentElement;
        const currentTheme = html.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        html.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
    });

    // Load saved theme preference
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        document.documentElement.setAttribute('data-theme', savedTheme);
    }

    // --- Animated Value Function ---
    const animateValue = (obj, start, end, duration, formatFn) => {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            obj.innerHTML = formatFn(progress * (end - start) + start);
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    };

    // --- Circular Progress Gauge Setter ---
    const setProgressRing = (percent) => {
        const radius = progressRingBar.r.baseVal.value;
        const circumference = radius * 2 * Math.PI;
        progressRingBar.style.strokeDasharray = `${circumference} ${circumference}`;
        
        const offset = circumference - (percent / 100) * circumference;
        progressRingBar.style.strokeDashoffset = offset;
    };

    // --- Render Prediction Results ---
    const renderResults = () => {
        if (!predictionResult) return;

        const isUSD = currentCurrency === 'USD';
        const priceOriginal = predictionResult.original_price_lakhs;
        const pricePredicted = predictionResult.predicted_price_lakhs;
        const priceDepreciation = predictionResult.depreciation_lakhs;
        const retentionPercent = predictionResult.value_retention_percentage;
        const age = predictionResult.estimated_age_years;

        // Show results containers
        resultsPlaceholder.classList.add('hide');
        resultsContent.classList.remove('hide');

        // Animate primary price display
        const targetValue = isUSD ? pricePredicted * USD_RATE : pricePredicted;
        const formatter = (v) => isUSD ? `$${Math.round(v).toLocaleString('en-US')}` : `₹${v.toFixed(2)} Lakhs`;
        
        animateValue(predictedPriceText, 0, targetValue, 1000, formatter);
        
        carAgeText.textContent = age;

        // Animate circular retention gauge
        setProgressRing(retentionPercent);
        animateValue(retentionPercentText, 0, retentionPercent, 1000, (v) => `${v.toFixed(1)}%`);

        // Update Breakdowns
        breakdownOriginal.textContent = formatCurrency(priceOriginal, currentCurrency);
        breakdownPredicted.textContent = formatCurrency(pricePredicted, currentCurrency);
        breakdownDepreciation.textContent = formatCurrency(priceDepreciation, currentCurrency);

        // Generate context-aware intelligence review
        let insight = "";
        const kms = parseInt(kmsDrivenInput.value);
        const engineSize = parseInt(engineInput.value);
        const power = parseInt(maxPowerInput.value);
        const mpg = parseFloat(mileageInput.value);

        if (retentionPercent >= 75) {
            insight = `Outstanding value retention! The vehicle preserves its equity extremely well. Large performance capacities (${power} bhp, ${engineSize} cc) paired with high fuel economy (${mpg} kmpl) maintain heavy demand in secondary marketplaces.`;
        } else if (age <= 3 && kms < 35000) {
            insight = `Highly premium spec. The low age (${age} years) and conservative driving distance (${kms.toLocaleString()} km) make this car highly competitive, indicating very minimal cosmetic or mechanical wear.`;
        } else if (retentionPercent < 45) {
            insight = `Heavy depreciation observed. High usage (${kms.toLocaleString()} km) or age (${age} years) has shifted the car into a budget utility category, but the price has now hit a very stable resale floor.`;
        } else {
            insight = `Healthy and standard valuation report. Driven ${kms.toLocaleString()} km with a solid ${engineSize} cc engine layout. The vehicle retains a solid ${retentionPercent.toFixed(1)}% of its theoretical showroom cost.`;
        }

        insightDescriptionText.textContent = insight;
    };

    // --- Form Submission / API Prediction ---
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Put button in loading state
        predictBtn.classList.add('loading');
        predictBtn.disabled = true;

        // Gather values
        const payload = {
            year: parseInt(yearInput.value),
            km_driven: parseInt(kmsDrivenInput.value),
            engine: parseFloat(engineInput.value),
            max_power: parseFloat(maxPowerInput.value),
            mileage: parseFloat(mileageInput.value),
            seats: parseFloat(form.querySelector('input[name="seats"]:checked').value),
            fuel_type: form.querySelector('input[name="fuel_type"]:checked').value,
            transmission: form.querySelector('input[name="transmission"]:checked').value,
            seller_type: form.querySelector('input[name="seller_type"]:checked').value,
            owner: form.querySelector('input[name="owner"]:checked').value
        };

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.error || 'Server error occurred during prediction.');
            }

            predictionResult = await response.json();
            renderResults();

            // Smooth scroll to results on mobile devices
            if (window.innerWidth <= 900) {
                document.getElementById('results-panel').scrollIntoView({ behavior: 'smooth' });
            }

        } catch (error) {
            console.error('Prediction Error:', error);
            alert(`Valuation failed: ${error.message}`);
        } finally {
            // Remove loading states
            predictBtn.classList.remove('loading');
            predictBtn.disabled = false;
        }
    });

    // Initialize displays
    yearVal.textContent = '2020 (6 yr old)';
    kmsDrivenVal.textContent = formatKms(45000);
    engineVal.textContent = '1200 cc';
    maxPowerVal.textContent = '85 bhp';
    mileageVal.textContent = '18.0 kmpl';
});
