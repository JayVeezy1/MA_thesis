const $del_dataset_modal = $('#confirm-delete-data')
const $del_account_modal = $('#confirm-delete-account')

// Charts
const quota_chart = createQuotaChart()


function formatBytes(bytes, decimals = 2) {
    if (bytes === 0) return '0 Bytes';

    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];

    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}


function createQuotaChart() {

    // Data
    const datasets = {
        labels: [],
        datasets: [{
            label: "Dataset quota",
            data: []
        }]
    }

    const config = {
        type: 'doughnut',
        data: datasets,
        options: {
            layout: {
                padding: 20
            },
            responsive: false,
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            let label = context.label || '';

                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed !== null) {
                                label += formatBytes(context.parsed)
                            }
                            return label;
                        }
                    }
                },
                doughnutLabel: {
                    labels: [{
                        text: '0%',
                    }, {
                        text: 'used',
                    }]
                }
            }
        }
    }


    return new Chart(
        document.getElementById('chart-quota'),
        config
    )
}


function updateQuotaChart(data) {
    // Get labels (datasets + free)
    const labels = Object.keys(data.quota_used)
    labels.push("free")
    const values = Object.values(data.quota_used)
    values.push(data.quota_free)

    // Colors
    let color_scheme = new ColorScheme
    color_scheme.from_hex('e67373')
        .scheme('triade')
        .distance(1)
        .variation('pastel')    // results in 12 colors
    let unordered = color_scheme.colors().map(s => '#' + s)        // prepend # for chartjs
    const colors = [
        unordered[0], unordered[4], unordered[8], unordered[1], unordered[5], unordered[9],
        unordered[2], unordered[6], unordered[10], unordered[3], unordered[7], unordered[11],
    ]

    // Update chart
    quota_chart.data.labels = labels
    quota_chart.data.datasets[0].data = values
    quota_chart.data.datasets[0].backgroundColor = colors

    // Update label (inner text) = percentage of quota
    const sum_values = values.reduce((pv, cv) => pv + cv, 0)        // contains quota_free, too
    let percentage = 100 * (sum_values - data.quota_free) / sum_values
    console.log(sum_values, data.quota_free, percentage )
    percentage = (Math.round(percentage * 100) / 100).toFixed(2)
    quota_chart.options.plugins.doughnutLabel.labels[0].text = percentage + '%'

    // Perform update
    quota_chart.update()
}


(function () {
    // Get and show quota in chart
    $.getJSON(quota_url, updateQuotaChart)

    // Show modal for datasets/account deletion confirmation (invalid submit -> invalid feedback)
    if (show_delete_dataset) {
        let modal = new bootstrap.Modal($del_dataset_modal.get(0), {})
        modal.show()
    }
    if (show_delete_account) {
        let modal = new bootstrap.Modal($del_account_modal.get(0), {})
        modal.show()
    }

})()