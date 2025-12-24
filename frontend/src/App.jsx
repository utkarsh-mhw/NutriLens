import React, { useState, useEffect } from 'react';
import { Camera, Zap, TrendingDown, Activity, Heart, AlertCircle, Github } from 'lucide-react';

import axios from 'axios';

const NutriLens = () => {
  const [view, setView] = useState('landing');
  const [selectedProduct, setSelectedProduct] = useState(null);
  const [analysisData, setAnalysisData] = useState(null);
  const [currentSlide, setCurrentSlide] = useState(0);
  const [loading, setLoading] = useState(false);
  const [recImages, setRecImages] = useState({});
  const GITHUB_REPO_URL = 'https://github.com/utkarsh-mhw/NutriLens';

  const isProduction = window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1';
  
  const API_BASE_URL = isProduction ? '/api' : 'http://localhost:5002/api';
  const API_ENDPOINT = isProduction ? '/analyze_precomputed' : '/analyze';

  const demoProducts = [
    { id: 850027959184, name: 'Mint Chocolate Cookie', code: 850027959184, category: 'Snacks' },
    { id: 70253469008, name: 'Enriched egg noodle product, medium egg noodles', code: 70253469008, category: 'Pasta' },
    { id: 77908000197, name: 'No 19 Rigatoni with Spring Water', code: 77908000197, category: 'Pasta' },
    { id: 11110128478, name: 'Spaghetti', code: 11110128478, category: 'Pasta' },
    { id: 21333225007, name: 'Chunk Light Tuna', code: 21333225007, category: 'Canned' },
    { id: 23547842018, name: 'Organic Traditional Whole Wheat Lomein', code: 23547842018, category: 'Pasta' },
    { id: 70038648604, name: 'Cream of chicken soup', code: 70038648604, category: 'Soup' },
    { id: 29737014012, name: 'Tortellini rosa', code: 29737014012, category: 'Pasta' },
    { id: 859924003020, name: "Leo's, ravioli, spinach & cheese", code: 859924003020, category: 'Pasta' },
    { id: 72036014689, name: 'Cheese ravioli', code: 72036014689, category: 'Pasta' },
    { id: 779566111719, name: 'Mini ravioli, 3 cheese', code: 779566111719, category: 'Pasta' },
    { id: 8006013990903, name: 'Eggplant parmesan ravioli', code: 8006013990903, category: 'Pasta' },
    { id: 856646004502, name: 'The original cold-pressed lemonade fruit juice drink blend', code: 856646004502, category: 'Beverages' },
    { id: 85239184417, name: 'Chunky Chicken Noodle Soup', code: 85239184417, category: 'Soup' },
    { id: 11110491985, name: 'Diet cola soda', code: 11110491985, category: 'Beverages' },
    { id: 86854036075, name: 'Diet Soda', code: 86854036075, category: 'Beverages' },
    { id: 896743002025, name: 'Recovery drink pomegranate punch', code: 896743002025, category: 'Beverages' },
    { id: 72554001628, name: 'Frozen dairy dessert cone', code: 72554001628, category: 'Frozen' },
    { id: 41789002373, name: 'Shrimp ramen noodle soup', code: 41789002373, category: 'Soup' },
  ];

  const slideImages = [
  { 
    image: '/demo_snapshots/confused_person.jpg', 
    text: 'Shopping healthy shouldn\'t be overwhelming',
    emoji: '🛒' // fallback
  },
  { 
    image: '/demo_snapshots/complex_ing.jpg', 
    text: 'Ingredient labels can be confusing',
    emoji: '🔬' // fallback
  },
  { 
    image: '/demo_snapshots/eat_clean.jpg', 
    text: 'Know exactly how processed your food is',
    emoji: '🥗' // fallback
  },
];

  const getProductImage = (code) => {
    return `/demo_snapshots/${code}.jpg`;
  };

  const getRecommendationImage = async (productCode) => {
    try {
      const response = await fetch(`https://world.openfoodfacts.org/api/v0/product/${productCode}.json`);
      const data = await response.json();
      
      if (data.status === 0) {
        return null; // Product not found
      }
      
      const imageUrl = 
        data.product?.image_small_url ||
        data.product?.image_thumb_url ||
        data.product?.image_url ||
        data.product?.image_front_small_url ||
        data.product?.image_front_thumb_url ||
        data.product?.image_front_url ||
        null;
      
      return imageUrl;
    } catch (error) {
      console.error(`Failed to fetch image for ${productCode}:`, error);
      return null;
    }
  };

  const analyzeProduct = async (productName) => {
    setLoading(true);
    setAnalysisData(null);
    setRecImages({});

    try {
      console.log(`Using ${isProduction ? 'PRODUCTION' : 'LOCAL'} API`);
      
      let data;
      
      if (isProduction) {
        // PRODUCTION: Load from JSON file
        // const response = await fetch('./demo_snapshots/precomputed_payloads.json');
        // const response = await fetch(`${import.meta.env.BASE_URL}demo_snapshots/precomputed_payloads.json`);
        // const response = await fetch('https://nutri-lens-nu.vercel.app/demo_snapshots/precomputed_payloads.json');
        // const allData = await response.json();
        // data = allData[productName];
        console.log('Fetching JSON...');
        const response = await fetch('https://nutri-lens-nu.vercel.app/demo_snapshots/precomputed_payloads.json');
        console.log('Response status:', response.status);
        const allData = await response.json();
        console.log('All data keys:', Object.keys(allData));
        console.log('Looking for product:', productName);
        data = allData[productName];
        console.log('Found data:', data);
      } else {
        // LOCAL: Use your ML backend
        const response = await axios.post('http://localhost:5002/api/analyze', {
          product_name: productName
        });
        
        data = response.data;
        
        if (typeof data === 'string') {
          data = data.replace(/:\s*NaN/g, ': null');
          data = JSON.parse(data);
        }
      }
      
      console.log("Getting response:", data);
      console.log("Success:", data.success);
      console.log("NOVA score:", data.nova_score);

      if (data && data.success) {
        const formattedData = {
          product_name: data.product_name,
          nova_score: data.nova_score,
          additives: data.product_macros.additives_n || 0,
          additive_density: data.product_macros.additive_density || 0,
          sugars: data.product_macros.sugars_100g || 0,
          fiber: data.product_macros.fiber_100g || 0,
          carbs: data.product_macros.carbohydrates_100g || 0,
          fat: (data.product_macros['saturated-fat_100g'] || 0) +
              (data.product_macros['monounsaturated-fat_100g'] || 0) +
              (data.product_macros['polyunsaturated-fat_100g'] || 0),
          ai_summary: data.explain_nova,
          recommendations: data.recommendations.map((rec) => ({
            name: rec[0],
            nova: rec[1],
            code: rec[2],
          })),
        };

        setAnalysisData(formattedData);
        setView('results');

        formattedData.recommendations.forEach(async (rec) => {
          const imageUrl = await getRecommendationImage(rec.code);
          setRecImages(prev => ({
            ...prev,
            [rec.code]: imageUrl
          }));
        });
      } else {
        alert(data?.error || 'Error analyzing product');
      }
    } catch (err) {
      console.error(err);
      alert('Failed to fetch data from backend');
    } finally {
      setLoading(false);
    }
  };

  const getNovaColor = (score) => {
    const colors = {
      1: 'bg-green-500',
      2: 'bg-yellow-400',
      3: 'bg-orange-500',
      4: 'bg-red-500',
    };
    return colors[score] || 'bg-gray-400';
  };

  const getNovaLabel = (score) => {
    const labels = {
      1: 'Unprocessed',
      2: 'Minimally Processed',
      3: 'Highly Processed',
      4: 'Very Highly Processed',
    };
    return labels[score] || 'Unknown';
  };

  React.useEffect(() => {
    const timer = setInterval(() => {
      setCurrentSlide((prev) => (prev + 1) % slideImages.length);
    }, 3000);
    return () => clearInterval(timer);
  }, []);
  const GitHubButton = () => (
    <a
      href={GITHUB_REPO_URL}
      target="_blank"
      rel="noopener noreferrer"
      className="fixed top-6 left-6 flex items-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-900 text-white rounded-full shadow-lg transition-all transform hover:scale-105 z-50"
    >
      <Github size={20} />
      <span className="font-medium">View Code</span>
    </a>
  );


  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-100">
        <div className="text-center">
          <div className="text-6xl mb-4 animate-spin">⏳</div>
          <p className="text-xl text-gray-700">Analyzing product, please wait...</p>
        </div>
      </div>
    );
  }

  if (view === 'landing') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 flex relative">
        
        {/* View Code Button */}
        <GitHubButton />

        <div className="w-1/2 flex flex-col justify-center items-start px-20">
          <div className="space-y-6">
            <h1 className="text-7xl font-bold text-gray-800 tracking-tight">
              NutriLens
            </h1>
            <p className="text-2xl text-gray-600 font-light">
              Eat Smart, Live Better
            </p>
            <button
              onClick={() => setView('selection')}
              className="mt-8 px-8 py-4 bg-blue-600 text-white rounded-full text-lg font-semibold hover:bg-blue-700 transition-all shadow-lg hover:shadow-xl transform hover:scale-105"
            >
              Analyze Product →
            </button>
            <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4 rounded-r-lg max-w-xl">
              <p className="text-sm text-gray-700">
                <strong>Academic Project:</strong> This tool uses data from Open Food Facts for educational purposes only.
                Not intended as medical or nutritional advice. Please consult healthcare professionals for dietary guidance.
              </p>
            </div>
          </div>
        </div>

        <div className="w-1/2 flex items-center justify-center bg-white">
          <div className="text-center transition-all duration-500 px-12">
            <div className="relative w-full h-96 mb-6 rounded-2xl overflow-hidden shadow-xl">
              <img
                src={slideImages[currentSlide].image}
                alt={slideImages[currentSlide].text}
                className="w-full h-full object-cover"
                onError={(e) => {
                  e.target.style.display = 'none';
                  e.target.nextElementSibling.style.display = 'flex';
                }}
              />
              <div className="absolute inset-0 hidden items-center justify-center text-9xl animate-pulse">
                {slideImages[currentSlide].emoji}
              </div>
            </div>
            <p className="text-3xl font-semibold text-gray-700">
              {slideImages[currentSlide].text}
            </p>
          </div>
        </div>
      </div>
    );
  }

  if (view === 'selection') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 flex">
        <div className="w-1/2 flex flex-col justify-center items-start px-20">
          <div className="space-y-6">
            <h1 className="text-7xl font-bold text-gray-800 tracking-tight">
              NutriLens
            </h1>
            <p className="text-2xl text-gray-600 font-light">
              Eat Smart, Live Better
            </p>
          </div>
        </div>
        <div className="w-1/2 flex items-center justify-center p-8">
          <div className="bg-white rounded-3xl shadow-2xl p-8 max-h-[90vh] overflow-y-auto w-full">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-3xl font-bold text-gray-800">Select a Product</h2>
              <button
                onClick={() => setView('landing')}
                className="text-gray-500 hover:text-gray-700 text-2xl"
              >
                ×
              </button>
            </div>
            <div className="grid grid-cols-3 gap-4">
              {demoProducts.map((product) => (
                <button
                  key={product.id}
                  onClick={() => {
                    setSelectedProduct(product);
                    analyzeProduct(product.name);
                  }}
                  className="bg-gray-50 hover:bg-blue-50 border-2 border-gray-200 hover:border-blue-400 rounded-xl p-4 transition-all transform hover:scale-105 overflow-hidden"
                >
                  <div className="relative w-full h-32 mb-2 bg-gray-100 rounded-lg overflow-hidden">
                    <img 
                      src={getProductImage(product.code)} 
                      alt={product.name}
                      className="w-full h-full object-cover"
                      onError={(e) => {
                        e.target.style.display = 'none';
                        e.target.nextElementSibling.style.display = 'flex';
                      }}
                    />
                    <div className="absolute inset-0 hidden items-center justify-center text-5xl">
                      🍽️
                    </div>
                  </div>
                  <p className="text-sm font-medium text-gray-700 line-clamp-2">{product.name}</p>
                  <p className="text-xs text-gray-500 mt-1">{product.category}</p>
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (view === 'results' && analysisData) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 flex">
        <div className="w-1/2 flex flex-col justify-center items-start px-20">
          <div className="space-y-6">
            <h1 className="text-7xl font-bold text-gray-800 tracking-tight">
              NutriLens
            </h1>
            <p className="text-2xl text-gray-600 font-light">
              Eat Smart, Live Better
            </p>
            <button
              onClick={() => setView('selection')}
              className="mt-8 px-6 py-3 bg-gray-600 text-white rounded-full text-base font-semibold hover:bg-gray-700 transition-all"
            >
              ← Choose Another
            </button>
          </div>
        </div>
        <div className="w-1/2 p-8 overflow-y-auto">
          <div className="bg-white rounded-3xl shadow-2xl p-8 space-y-6">
            <div className="text-center border-b pb-6">
              <div className="relative w-40 h-40 mx-auto mb-4 bg-gray-100 rounded-xl overflow-hidden">
                <img 
                  src={getProductImage(selectedProduct.code)} 
                  alt={selectedProduct.name}
                  className="w-full h-full object-cover"
                  onError={(e) => {
                    e.target.style.display = 'none';
                    e.target.nextElementSibling.style.display = 'flex';
                  }}
                />
                <div className="absolute inset-0 hidden items-center justify-center text-6xl">
                  🍽️
                </div>
              </div>
              <h2 className="text-2xl font-bold text-gray-800">{analysisData.product_name}</h2>
            </div>
            <div className={`${getNovaColor(analysisData.nova_score)} text-white rounded-2xl p-6 text-center`}>
              <div className="text-5xl font-bold mb-2">NOVA Score: {analysisData.nova_score}</div>
              <div className="text-lg font-medium">{getNovaLabel(analysisData.nova_score)}</div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-red-50 border-2 border-red-200 rounded-xl p-4">
                <div className="flex items-center justify-between">
                  <AlertCircle className="text-red-500" size={24} />
                  <span className="text-2xl font-bold text-red-600">{analysisData.additives}</span>
                </div>
                <p className="text-sm text-gray-600 mt-2">Number of Additives</p>
              </div>
              
              <div className="bg-yellow-50 border-2 border-yellow-200 rounded-xl p-4">
                <div className="flex items-center justify-between">
                  <Activity className="text-yellow-600" size={24} />
                  <span className="text-2xl font-bold text-yellow-600">{analysisData.sugars}g</span>
                </div>
                <p className="text-sm text-gray-600 mt-2">Sugars / 100g</p>
              </div>
              <div className="bg-green-50 border-2 border-green-200 rounded-xl p-4">
                <div className="flex items-center justify-between">
                  <Heart className="text-green-600" size={24} />
                  <span className="text-2xl font-bold text-green-600">{analysisData.fiber}g</span>
                </div>
                <p className="text-sm text-gray-600 mt-2">Fiber / 100g</p>
              </div>
              <div className="bg-blue-50 border-2 border-blue-200 rounded-xl p-4">
                <div className="flex items-center justify-between">
                  <Heart className="text-blue-600" size={24} />
                  <span className="text-2xl font-bold text-blue-600">{analysisData.carbs}g</span>
                </div>
                <p className="text-sm text-gray-600 mt-2">Carbs / 100g</p>
              </div>
            </div>
            <div className="bg-purple-50 border-2 border-purple-200 rounded-xl p-4">
              <p className="text-sm text-gray-700 italic">
                💡 {analysisData.ai_summary}
              </p>
            </div>
            <div>
              <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
                <TrendingDown className="mr-2 text-green-600" />
                Less Processed Alternatives
              </h3>
              <div className="space-y-3">
                {analysisData.recommendations.map((rec, idx) => (
                  <div key={idx} className="bg-green-50 border-2 border-green-200 rounded-xl p-4 flex items-center justify-between hover:bg-green-100 transition-all">
                    <div className="flex items-center space-x-4">
                      <div className="relative w-16 h-16 bg-gray-100 rounded-lg overflow-hidden flex-shrink-0">
                        {recImages[rec.code] ? (
                          <img 
                            src={recImages[rec.code]} 
                            alt={rec.name}
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          <div className="w-full h-full flex items-center justify-center text-2xl">
                            {recImages[rec.code] === null ? '🍽️' : '⏳'}
                          </div>
                        )}
                      </div>
                      <div>
                        <p className="font-semibold text-gray-800">{rec.name}</p>
                        <p className="text-sm text-green-600">NOVA Score: {rec.nova}</p>
                      </div>
                    </div>
                    <div className="bg-green-500 text-white px-3 py-1 rounded-full text-sm font-semibold">
                      Better
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return null;
};

export default NutriLens;