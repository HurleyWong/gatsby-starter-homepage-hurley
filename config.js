module.exports = {
    pathPrefix: '',
    siteUrl: 'https://hurleyhuang.netlify.app/',
    siteTitle: 'Hurley Huang',
    siteDescription: '',
    author: 'Hurley',
    postsForArchivePage: 3,
    defaultLanguage: 'en',
    disqusScript: process.env.DISQUS_SCRIPT || 'https://hurleyjames.disqus.com/embed.js',
    pages: {
        home: '/',
        blog: 'blog',
        contact: 'contact',
        resume: 'resume',
        tag: 'tags',
    },
    social: {
        github: 'https://github.com/HurleyJames',
        facebook: 'https://www.facebook.com/profile.php?id=100014949587803',
        twitter: 'https://twitter.com/HurleyHuang23',
        instagram: 'https://www.instagram.com/hurleyhuang/',
        rss: '/rss.xml',
    },
    contactFormUrl: process.env.CONTACT_FORM_ENDPOINT || 'https://getform.io/f/75f0d574-50f0-4d68-a667-8112e6e70f73',
    googleAnalyticTrackingId: process.env.GA_TRACKING_ID || '',
    tags: {
        leetcode: {
            name: 'leetcode',
            description: 'LeetCode is a website where people–mostly software engineers–practice their coding skills.',
            color: '#44566c',
        },
        JavaScript: {
            name: 'JavaScript',
            description: 'JavaScript is an object-oriented programming language used alongside HTML and CSS to give functionality to web pages.',
            color: '#44566c',
        },
        Nodejs: {
            name: 'Node.js',
            description: 'Node.js is a tool for executing JavaScript in a variety of environments.',
            color: '#44566c',
        },
        TypeScript: {
            name: 'TypeScript',
            description: 'TypeScript is a typed superset of JavaScript that compiles to plain JavaScript.',
            color: '#44566c',
        },
        React: {
            name: 'React',
            description: 'React is an open source JavaScript library used for designing user interfaces.',
            color: '#44566c',
        },
        html: {
            name: 'HTML',
            description: 'A markup language that powers the web. All websites use HTML for structuring the content.',
            color: '#44566c',
        },
        css: {
            name: 'css',
            description: 'CSS is used to style the HTML element and to give a very fancy look for the web application.',
            color: '#44566c',
        },
        Python: {
            name: 'Python',
            description: 'A general purpose programming language that is widely used for developing various applications.',
            color: '#44566c',
        },
        Android: {
            name: 'Android',
            description: 'Android is a mobile operating system, designed primarily for touchscreen mobile devices.',
            color: '#44566c',
        },
        Java: {
            name: 'Java',
            description: 'Java is a general-purpose programming language that is class-based, object-oriented.',
            color: '#44566c',
        },
        Kotlin: {
            name: 'Kotlin',
            description: 'Kotlin is a cross-platform, statically typed, general-purpose programming language with type inference.',
            color: '#44566c',
        },
        Blockchain: {
            name: 'Blockchain',
            description: 'A blockchain, is a growing list of records, called blocks, that are linked using cryptography.',
            color: '#44566c',
        },
        AI: {
            name: 'AI',
            description: 'Artificial intelligence is intelligence demonstrated by machines, unlike the natural intelligence displayed by humans and animals.',
            color: '#44566c',
        },
        BigData: {
            name: 'BigData',
            description: 'Big data is a field that treats ways to analyze, systematically extract information from.',
            color: '#44566c',
        },
        CloudComputing: {
            name: 'CloudComputing',
            description: 'Cloud computing is the on-demand availability of computer system resources, especially data storage and computing power.',
            color: '#44566c',
        },
    },
};