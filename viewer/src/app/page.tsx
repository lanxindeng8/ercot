import Link from "next/link";
import PriceTable from "@/components/PriceTable";
import PriceChart from "@/components/PriceChart";

export default function Home() {
  return (
    <div className="container">
      <header>
        <h1>ERCOT Settlement Point Prices</h1>
        <p>Data sourced from ERCOT Public API</p>
        <nav className="main-nav">
          <span className="main-nav-link disabled">Dashboard</span>
          <Link href="/" className="main-nav-link active">Market Data</Link>
          <span className="main-nav-link disabled">Predictions</span>
          <span className="main-nav-link disabled">Dispatch</span>
        </nav>
      </header>

      <PriceChart />

      <div className="tables-grid">
        <PriceTable
          title="Real-Time Settlement Point Prices Display"
          apiEndpoint="/api/rtm-spp"
          timeColumnLabel="Interval Ending"
        />

        <PriceTable
          title="DAM Settlement Point Prices Display"
          apiEndpoint="/api/dam-spp"
          timeColumnLabel="Hour Ending"
        />
      </div>

      <footer>
        <p>
          SPP values include the Real-Time Reliability Deployment Price Adders.
          <br />
          Data from{" "}
          <a
            href="https://www.ercot.com"
            target="_blank"
            rel="noopener noreferrer"
          >
            ERCOT
          </a>
        </p>
      </footer>
    </div>
  );
}
