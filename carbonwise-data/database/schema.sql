PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS dataset_versions (
    dataset_version_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('processing', 'accepted', 'partial', 'rejected')),
    source_id TEXT,
    orders_row_count INTEGER NOT NULL DEFAULT 0 CHECK (orders_row_count >= 0),
    plants_row_count INTEGER NOT NULL DEFAULT 0 CHECK (plants_row_count >= 0),
    transport_duties_row_count INTEGER NOT NULL DEFAULT 0 CHECK (transport_duties_row_count >= 0),
    capacity_events_row_count INTEGER NOT NULL DEFAULT 0 CHECK (capacity_events_row_count >= 0),
    validation_status TEXT NOT NULL CHECK (validation_status IN ('pending', 'valid', 'invalid')),
    summary TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS orders (
    dataset_version_id TEXT NOT NULL,
    row_order INTEGER NOT NULL CHECK (row_order >= 0),
    order_id TEXT NOT NULL,
    product TEXT NOT NULL,
    qty INTEGER NOT NULL CHECK (qty > 0),
    destination TEXT NOT NULL,
    due TEXT NOT NULL,
    PRIMARY KEY (dataset_version_id, order_id),
    UNIQUE (dataset_version_id, row_order),
    FOREIGN KEY (dataset_version_id) REFERENCES dataset_versions(dataset_version_id)
        ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS plants (
    dataset_version_id TEXT NOT NULL,
    row_order INTEGER NOT NULL CHECK (row_order >= 0),
    plant TEXT NOT NULL,
    product TEXT NOT NULL,
    weekly_capacity INTEGER NOT NULL CHECK (weekly_capacity > 0),
    unit_cost REAL NOT NULL CHECK (unit_cost >= 0),
    energy_kwh_per_unit REAL NOT NULL CHECK (energy_kwh_per_unit >= 0),
    carbon_kg_per_kwh REAL NOT NULL CHECK (carbon_kg_per_kwh >= 0),
    PRIMARY KEY (dataset_version_id, plant, product),
    UNIQUE (dataset_version_id, row_order),
    FOREIGN KEY (dataset_version_id) REFERENCES dataset_versions(dataset_version_id)
        ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS transport_duties (
    dataset_version_id TEXT NOT NULL,
    row_order INTEGER NOT NULL CHECK (row_order >= 0),
    plant TEXT NOT NULL,
    destination TEXT NOT NULL,
    transport_cost_per_unit REAL NOT NULL CHECK (transport_cost_per_unit >= 0),
    transit_days INTEGER NOT NULL CHECK (transit_days >= 0),
    duty_pct REAL NOT NULL CHECK (duty_pct BETWEEN 0 AND 1),
    carbon_kg_per_unit REAL NOT NULL CHECK (carbon_kg_per_unit >= 0),
    PRIMARY KEY (dataset_version_id, plant, destination),
    UNIQUE (dataset_version_id, row_order),
    FOREIGN KEY (dataset_version_id) REFERENCES dataset_versions(dataset_version_id)
        ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS capacity_events (
    dataset_version_id TEXT NOT NULL,
    row_order INTEGER NOT NULL CHECK (row_order >= 0),
    event_id TEXT NOT NULL,
    plant TEXT NOT NULL,
    from_date TEXT NOT NULL,
    to_date TEXT NOT NULL,
    capacity_loss_pct REAL NOT NULL CHECK (capacity_loss_pct BETWEEN 0 AND 1),
    PRIMARY KEY (dataset_version_id, event_id),
    UNIQUE (dataset_version_id, row_order),
    FOREIGN KEY (dataset_version_id) REFERENCES dataset_versions(dataset_version_id)
        ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS ingestion_logs (
    ingestion_log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_version_id TEXT,
    created_at TEXT NOT NULL,
    severity TEXT NOT NULL CHECK (severity IN ('info', 'warning', 'error')),
    table_name TEXT,
    row_id TEXT,
    rule_name TEXT NOT NULL,
    message TEXT NOT NULL,
    FOREIGN KEY (dataset_version_id) REFERENCES dataset_versions(dataset_version_id)
        ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_orders_version_due
    ON orders(dataset_version_id, due);
CREATE INDEX IF NOT EXISTS idx_plants_version_product
    ON plants(dataset_version_id, product);
CREATE INDEX IF NOT EXISTS idx_transport_version_destination
    ON transport_duties(dataset_version_id, destination);
CREATE INDEX IF NOT EXISTS idx_capacity_events_version_plant
    ON capacity_events(dataset_version_id, plant);
CREATE INDEX IF NOT EXISTS idx_ingestion_logs_version_severity
    ON ingestion_logs(dataset_version_id, severity);
