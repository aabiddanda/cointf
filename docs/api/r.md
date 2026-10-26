# R API

Reference for the R package in `r/`, generated from its help pages (the same text as
`?loglik_fast` and friends in R) by `docs/api/make_r_reference.R`. Each function calls the
Python function of the same name; see [Using cointf from R](../r.md) for setup.

## loglik_fast()

```{raw} html
<span id="r-loglik_fast"></span>
<p><strong>Log-likelihood of crossover positions under the Housworth-Stahl model</strong></p><span id='topic+loglik_fast'></span>

<h3>Description</h3>

<p>Crossovers come from an interference pathway (a stationary renewal process
with gamma-mixture gaps governed by <code>nu</code>) and an escape pathway (a Poisson
process carrying a fraction <code>p</code> of crossovers). The likelihood sums over
the unobserved pathway labels exactly.
</p>


<h3>Usage</h3>

<pre><code class='language-R'>loglik_fast(data, L, nu, p, kernel = NULL, per_meiosis = FALSE)
</code></pre>


<h3>Arguments</h3>

<table role = "presentation">
<tr><td><code id="loglik_fast_+3A_data">data</code></td>
<td>
<p>A list with one numeric vector of crossover positions (in
Morgans) per meiosis, or an object from <a href="#r-meioses"><code>meioses()</code></a>.</p>
</td></tr>
<tr><td><code id="loglik_fast_+3A_L">L</code></td>
<td>
<p>Chromosome length in Morgans: one value or one per meiosis. Must
be <code>NULL</code> when <code>data</code> comes from <a href="#r-meioses"><code>meioses()</code></a>.</p>
</td></tr>
<tr><td><code id="loglik_fast_+3A_nu">nu</code></td>
<td>
<p>Interference parameter, <code style="white-space: pre;">&#8288;&gt; 0&#8288;</code>: one value or one per meiosis.
<code>nu = 1</code> is no interference.</p>
</td></tr>
<tr><td><code id="loglik_fast_+3A_p">p</code></td>
<td>
<p>Fraction of crossovers escaping interference, in <code style="white-space: pre;">&#8288;[0, 1]&#8288;</code>: one
value or one per meiosis.</p>
</td></tr>
<tr><td><code id="loglik_fast_+3A_kernel">kernel</code></td>
<td>
<p>Optional kernel, e.g. from <a href="#r-chebyshev_kernel"><code>chebyshev_kernel()</code></a>; the default
is the exact series.</p>
</td></tr>
<tr><td><code id="loglik_fast_+3A_per_meiosis">per_meiosis</code></td>
<td>
<p>If <code>TRUE</code>, return one log-likelihood per meiosis
instead of the sum.</p>
</td></tr>
</table>


<h3>Value</h3>

<p>The total log-likelihood, or a numeric vector with one value per
meiosis if <code>per_meiosis = TRUE</code>.
</p>


<h3>Examples</h3>

<pre><code class='language-R'>data &lt;- simulate_meioses(L = 1.5, nu = 4, p = 0.05, n = 200, seed = 7)
loglik_fast(data, 1.5, nu = 4, p = 0.05)

</code></pre>

```

## loglik_quartet()

```{raw} html
<span id="r-loglik_quartet"></span>
<p><strong>Log-likelihood of phase-unknown (quartet) crossover data</strong></p><span id='topic+loglik_quartet'></span>

<h3>Description</h3>

<p>For a parent in a family with two offspring and no grandparents, each
chromosome yields the pooled crossovers of two meioses, without knowing
which meiosis each crossover came from. The likelihood sums over every
assignment of crossovers to the two meioses or to the escape pathway.
</p>


<h3>Usage</h3>

<pre><code class='language-R'>loglik_quartet(data, L, nu, p, kernel = NULL, per_meiosis = FALSE)
</code></pre>


<h3>Arguments</h3>

<table role = "presentation">
<tr><td><code id="loglik_quartet_+3A_data">data</code></td>
<td>
<p>A list with one numeric vector of pooled crossover positions
(in Morgans) per quartet, or an object from <a href="#r-meioses"><code>meioses()</code></a>.</p>
</td></tr>
<tr><td><code id="loglik_quartet_+3A_L">L</code></td>
<td>
<p>Chromosome length in Morgans: one value or one per meiosis. Must
be <code>NULL</code> when <code>data</code> comes from <a href="#r-meioses"><code>meioses()</code></a>.</p>
</td></tr>
<tr><td><code id="loglik_quartet_+3A_nu">nu</code>, <code id="loglik_quartet_+3A_p">p</code></td>
<td>
<p>As for <a href="#r-loglik_fast"><code>loglik_fast()</code></a>; both meioses of a quartet share them.</p>
</td></tr>
<tr><td><code id="loglik_quartet_+3A_kernel">kernel</code></td>
<td>
<p>Optional kernel, e.g. from <a href="#r-chebyshev_kernel"><code>chebyshev_kernel()</code></a>; the default
is the exact series.</p>
</td></tr>
<tr><td><code id="loglik_quartet_+3A_per_meiosis">per_meiosis</code></td>
<td>
<p>If <code>TRUE</code>, return one value per quartet (one entry of
<code>data</code>) instead of the sum. The name matches <a href="#r-loglik_fast"><code>loglik_fast()</code></a>.</p>
</td></tr>
</table>


<h3>Value</h3>

<p>The total log-likelihood, or a numeric vector with one value per
quartet if <code>per_meiosis = TRUE</code>.
</p>


<h3>Examples</h3>

<pre><code class='language-R'>quartets &lt;- simulate_quartets(L = 1.5, nu = 4, p = 0.05, n = 200, seed = 7)
loglik_quartet(quartets, 1.5, nu = 4, p = 0.05)

</code></pre>

```

## meioses()

```{raw} html
<span id="r-meioses"></span>
<p><strong>Prepare crossover data for repeated likelihood evaluation</strong></p><span id='topic+meioses'></span>

<h3>Description</h3>

<p>Validates and sorts the positions, groups meioses by crossover count and
precomputes every distance the likelihood needs. Build it once and pass it
to <a href="#r-loglik_fast"><code>loglik_fast()</code></a> or <a href="#r-loglik_quartet"><code>loglik_quartet()</code></a> (with <code>L = NULL</code>) for each
parameter value, for example inside an optimiser.
</p>


<h3>Usage</h3>

<pre><code class='language-R'>meioses(data, L)
</code></pre>


<h3>Arguments</h3>

<table role = "presentation">
<tr><td><code id="meioses_+3A_data">data</code></td>
<td>
<p>A list with one numeric vector of crossover positions (in
Morgans) per meiosis; an entry may be empty (<code>numeric(0)</code>). For quartet
data, one entry per quartet.</p>
</td></tr>
<tr><td><code id="meioses_+3A_L">L</code></td>
<td>
<p>Chromosome length in Morgans: one value, or one per entry of
<code>data</code>.</p>
</td></tr>
</table>


<h3>Value</h3>

<p>A Python <code>Meioses</code> object.
</p>


<h3>Examples</h3>

<pre><code class='language-R'>mz &lt;- meioses(list(c(0.2, 0.9), numeric(0), 0.5), L = 1.2)
loglik_fast(mz, NULL, nu = 4, p = 0.05)

</code></pre>

```

## chebyshev_kernel()

```{raw} html
<span id="r-chebyshev_kernel"></span>
<p><strong>Faster approximate likelihood kernel</strong></p><span id='topic+chebyshev_kernel'></span>

<h3>Description</h3>

<p>Piecewise Chebyshev fits of the log gap density and log survival function,
built once per value of <code>nu</code> and cached in the returned object. Keep one
kernel and reuse it across calls so the cache is reused. With a shared
<code>nu</code> it is 6-10 times faster than the exact series, with per-meiosis
log-likelihood differences around 1e-13.
</p>


<h3>Usage</h3>

<pre><code class='language-R'>chebyshev_kernel(
  tol = 1e-12,
  u_min = 1e-06,
  u_max = 5,
  u_split = 0.25,
  n_nodes = 16,
  max_distinct = 16,
  max_cached = 256
)
</code></pre>


<h3>Arguments</h3>

<table role = "presentation">
<tr><td><code id="chebyshev_kernel_+3A_tol">tol</code></td>
<td>
<p>Acceptance threshold for each fitted piece (not an error bound).</p>
</td></tr>
<tr><td><code id="chebyshev_kernel_+3A_u_min">u_min</code>, <code id="chebyshev_kernel_+3A_u_max">u_max</code></td>
<td>
<p>Distances (on the interference scale) inside this range
use the fit; the rest use the exact series.</p>
</td></tr>
<tr><td><code id="chebyshev_kernel_+3A_u_split">u_split</code></td>
<td>
<p>Boundary between pieces fitted in log distance and in
distance.</p>
</td></tr>
<tr><td><code id="chebyshev_kernel_+3A_n_nodes">n_nodes</code></td>
<td>
<p>Chebyshev nodes per piece.</p>
</td></tr>
<tr><td><code id="chebyshev_kernel_+3A_max_distinct">max_distinct</code></td>
<td>
<p>Largest number of distinct <code>nu</code> values in one call for
which fits are built; above it the exact series is used.</p>
</td></tr>
<tr><td><code id="chebyshev_kernel_+3A_max_cached">max_cached</code></td>
<td>
<p>Fits kept before the cache is cleared.</p>
</td></tr>
</table>


<h3>Value</h3>

<p>A Python <code>ChebyshevKernel</code> object, to pass as <code>kernel</code>.
</p>


<h3>Examples</h3>

<pre><code class='language-R'>kernel &lt;- chebyshev_kernel()
data &lt;- simulate_meioses(L = 1.5, nu = 4, p = 0.05, n = 200, seed = 7)
loglik_fast(data, 1.5, nu = 4, p = 0.05, kernel = kernel)

</code></pre>

```

## simulate_meioses(), simulate_quartets()

```{raw} html
<span id="r-simulate_meioses"></span>
<p><strong>Simulate crossover positions under the Housworth-Stahl model</strong></p><span id='topic+simulate_meioses'></span><span id='topic+simulate_quartets'></span>

<h3>Description</h3>

<p><a href="#r-simulate_meioses"><code>simulate_meioses()</code></a> draws independent meioses. <a href="#r-simulate_meioses"><code>simulate_quartets()</code></a>
draws phase-unknown data: each entry pools the crossovers of two
independent meioses.
</p>


<h3>Usage</h3>

<pre><code class='language-R'>simulate_meioses(L, nu, p, n, seed)

simulate_quartets(L, nu, p, n, seed)
</code></pre>


<h3>Arguments</h3>

<table role = "presentation">
<tr><td><code id="simulate_meioses_+3A_L">L</code></td>
<td>
<p>Chromosome length in Morgans, <code style="white-space: pre;">&#8288;&gt; 0&#8288;</code>.</p>
</td></tr>
<tr><td><code id="simulate_meioses_+3A_nu">nu</code></td>
<td>
<p>Interference parameter, <code style="white-space: pre;">&#8288;&gt; 0&#8288;</code>.</p>
</td></tr>
<tr><td><code id="simulate_meioses_+3A_p">p</code></td>
<td>
<p>Fraction of crossovers escaping interference, in <code style="white-space: pre;">&#8288;[0, 1]&#8288;</code>.</p>
</td></tr>
<tr><td><code id="simulate_meioses_+3A_n">n</code></td>
<td>
<p>Number of meioses (or quartets).</p>
</td></tr>
<tr><td><code id="simulate_meioses_+3A_seed">seed</code></td>
<td>
<p>Random seed, <code style="white-space: pre;">&#8288;&gt; 0&#8288;</code>. The same seed gives the same data as the
Python package.</p>
</td></tr>
</table>


<h3>Value</h3>

<p>A list of sorted numeric vectors of crossover positions in Morgans.
</p>


<h3>Examples</h3>

<pre><code class='language-R'>simulate_meioses(L = 1.5, nu = 4, p = 0.05, n = 3, seed = 7)
simulate_quartets(L = 1.5, nu = 4, p = 0.05, n = 3, seed = 7)

</code></pre>

```

## meiosis_list()

```{raw} html
<span id="r-meiosis_list"></span>
<p><strong>Crossover positions from long-format data</strong></p><span id='topic+meiosis_list'></span>

<h3>Description</h3>

<p>Turns one-row-per-crossover data into the list format the likelihood
functions take. Meioses without crossovers are kept by giving them a row
with <code>NA</code> position.
</p>


<h3>Usage</h3>

<pre><code class='language-R'>meiosis_list(position, meiosis)
</code></pre>


<h3>Arguments</h3>

<table role = "presentation">
<tr><td><code id="meiosis_list_+3A_position">position</code></td>
<td>
<p>Crossover positions in Morgans; <code>NA</code> for a meiosis without
crossovers.</p>
</td></tr>
<tr><td><code id="meiosis_list_+3A_meiosis">meiosis</code></td>
<td>
<p>Meiosis identifier for each row.</p>
</td></tr>
</table>


<h3>Value</h3>

<p>A named list with one numeric vector per meiosis, in order of
first appearance.
</p>


<h3>Examples</h3>

<pre><code class='language-R'>meiosis_list(c(0.2, 0.9, NA, 0.5), c("a", "a", "b", "c"))
</code></pre>

```

## cointf_python()

```{raw} html
<span id="r-cointf_python"></span>
<p><strong>The underlying Python module</strong></p><span id='topic+cointf_python'></span>

<h3>Description</h3>

<p>Gives direct access to the Python 'cointf' module, for functions without
an R wrapper (for example <code>cointf_python()$naive$loglik_quartet_naive</code>).
</p>


<h3>Usage</h3>

<pre><code class='language-R'>cointf_python()
</code></pre>


<h3>Value</h3>

<p>The 'cointf' Python module, as a reticulate object.
</p>

```

## Package overview and Python setup

```{raw} html
<span id="r-cointf-package"></span>
<p><strong>cointf: Housworth-Stahl likelihoods from R</strong></p><span id='topic+cointf'></span><span id='topic+cointf-package'></span>

<h3>Description</h3>

<p>An R interface to the Python package 'cointf'. Each R function calls the
Python function of the same name through 'reticulate', so results are
identical to the Python package.
</p>


<h3>Python environment</h3>

<p>When the package is loaded it declares its Python requirement with
<code>reticulate::py_require()</code>, and reticulate installs it into a managed
environment the first time Python is used. The requirement defaults to the
GitHub repository and can be changed with the option <code>cointf.python</code>, set
before loading the package, e.g. to install from the <code>dev</code> branch:
</p>
<div class="sourceCode r"><pre>options(cointf.python = "cointf @ git+https://github.com/aabiddanda/cointf@dev")
</pre></div>
<p>To use an existing Python environment instead (for example one where
'cointf' was installed with <code style="white-space: pre;">&#8288;pip install .&#8288;</code> from a clone), point
reticulate at it before loading the package, e.g.
<code>Sys.setenv(RETICULATE_PYTHON = "/path/to/venv/bin/python")</code>. The GitHub
requirement is then not declared. Setting <code>options(cointf.python = NULL)</code>
also skips it, for environments selected another way (e.g.
<code>reticulate::use_virtualenv()</code>).
</p>


<h3>Author(s)</h3>

<p><strong>Maintainer</strong>: Arjun Biddanda <a href="mailto:aabiddanda@gmail.com">aabiddanda@gmail.com</a>
</p>
<p>Authors:
</p>

<ul>
<li><p> Arjun Biddanda <a href="mailto:aabiddanda@gmail.com">aabiddanda@gmail.com</a>
</p>
</li></ul>



<h3>See Also</h3>

<p>Useful links:
</p>

<ul>
<li> <p><a href="https://github.com/aabiddanda/cointf">https://github.com/aabiddanda/cointf</a>
</p>
</li>
<li><p> Report bugs at <a href="https://github.com/aabiddanda/cointf/issues">https://github.com/aabiddanda/cointf/issues</a>
</p>
</li></ul>


```

