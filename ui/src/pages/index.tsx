export default function Home() {
  const checkAPI = async () => {
    const res = await fetch("http://127.0.0.1:8000/health");
    const data = await res.json();
    alert(JSON.stringify(data));
  };

  return (
    <main style={{ fontFamily: "monospace", padding: "2rem" }}>
      <h1>TID-AD-ASTRA 🌌</h1>
      <p>Backend connectivity test</p>
      <button onClick={checkAPI}>Ping API</button>
    </main>
  );
}

