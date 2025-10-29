import numpy as np
from numpy.linalg import inv, norm

class GGHCryptosystem:
    """
    Implementation of the GGH/HNF (Goldreich-Goldwasser-Halevi) cryptosystem.
    This is a lattice-based public-key cryptosystem.
    """
    
    def __init__(self, n=5, bound=4):
        """
        Initialize the cryptosystem.
        
        Args:
            n: Dimension of the lattice
            bound: Bound for generating "good" basis vectors
        """
        self.n = n
        self.bound = bound
        self.private_key = None  # "Good" basis (short, nearly orthogonal)
        self.public_key = None   # "Bad" basis (long, not orthogonal)
        
    def generate_good_basis(self):
        """
        Generate a "good" basis: short vectors that are nearly orthogonal.
        This is the private key.
        """
        # Generate random short vectors
        # These are rows of vectors, where each row represents an endpoint in n-dimensional space
        # We only need the end point coordinate, since we assume the start point
        # is the origin [0, 0, ..., 0]
        R = np.random.randint(-self.bound, self.bound + 1, size=(self.n, self.n))

        # Form lecture notes: Goldreich, Goldwasser and Halevi proposition
        k = int(np.ceil(np.sqrt(self.n))) * self.bound
        identity = np.eye(self.n)
        R = k * identity + R
        
        self.private_key = R.astype(float)
        print("Private key (good basis) generated:")
        print(self.private_key)
        print(f"Hadamard ratio: {self.hadamard_ratio(self.private_key):.4f}")
        
    def hadamard_ratio(self, basis):
        """
        Compute Hadamard ratio to measure basis quality.
        Closer to 1 means more orthogonal (better basis).
        """
        det = abs(np.linalg.det(basis))
        product_norms = np.prod([norm(basis[i]) for i in range(len(basis))])
        if product_norms == 0:
            return 0
        return (det / product_norms) ** (1 / len(basis))
    
    def generate_unimodular_matrix(self):
        """
        Generate a random unimodular matrix (determinant ±1).
        This will be used to transform the good basis into a bad one.
        """
        # Start with identity
        U = np.eye(self.n, dtype=int)
        
        # Apply random elementary row operations
        for _ in range(self.n * 3):
            i, j = np.random.choice(self.n, 2, replace=False)
            k = np.random.randint(-3, 4)
            # Add k times row j to row i
            U[i] += k * U[j]
        
        return U
    
    def hermite_normal_form(self, basis):
        """
        Compute the Hermite Normal Form (HNF) of a basis matrix.
        Uses integer Gaussian elimination to produce a lower triangular matrix.
        
        Returns:
            HNF matrix (lower triangular with positive diagonal)
        """
        # Work with integer matrix
        H = np.round(basis).astype(int)
        n = len(H)
        
        # Column-wise reduction (for lower triangular HNF)
        for j in range(n):
            # Find pivot (non-zero entry in column j at or below diagonal)
            pivot_row = j
            while pivot_row < n and H[pivot_row, j] == 0:
                pivot_row += 1
            
            if pivot_row >= n:
                continue  # Column is zero, skip
            
            # Swap rows if needed
            if pivot_row != j:
                H[[j, pivot_row]] = H[[pivot_row, j]]
            
            # Make diagonal positive
            if H[j, j] < 0:
                H[j] = -H[j]
            
            # Reduce entries below diagonal
            for i in range(j + 1, n):
                if H[i, j] != 0:
                    q = H[i, j] // H[j, j]
                    H[i] = H[i] - q * H[j]
            
            # Reduce entries above diagonal to be in range [0, H[j,j])
            for i in range(j):
                if H[i, j] != 0:
                    q = H[i, j] // H[j, j]
                    H[i] = H[i] - q * H[j]
        
        return H.astype(float)

    def generate_bad_basis(self):
        """
        Generate the public key as the Hermite Normal Form of the private key.
        This creates a "bad" basis (lower triangular, large entries).
        """
        if self.private_key is None:
            raise ValueError("Private key must be generated first")
        
        # Compute HNF
        self.public_key = self.hermite_normal_form(self.private_key)
        
        print("\nPublic key (HNF basis) generated:")
        print(self.public_key)
        print(f"Hadamard ratio: {self.hadamard_ratio(self.public_key):.4f}")
    
    def generate_keys(self):
        """Generate both private and public keys."""
        self.generate_good_basis()
        self.generate_bad_basis()
        
    def encrypt(self, message):
        """
        Encrypt a message vector using HNF-based encryption.
        """
        if self.public_key is None:
            raise ValueError("Keys must be generated first")
        
        ciphertext = ""
        for i in range(self.n):
            ciphertext = message - (message[i] // self.public_key[i, i]) * self.public_key[i]
        
        print(f"\nMessage (noise vector r): {message}")
        print(f"Ciphertext: {ciphertext}")
        
        return ciphertext
    
    def decrypt(self, ciphertext):
        """
        Decrypt a ciphertext using the private key.
        """
        if self.private_key is None:
            raise ValueError("Private key must be generated first")

        decrypted = ""
        aux = ciphertext @ inv(self.private_key)
        decrypted = ciphertext - np.array([round(ele) for ele in aux]) @ self.private_key
        
        print(f"\nDecrypted message: {decrypted}")
        
        return decrypted
    
    def LLL(self, basis, delta=0.75):
        # Initialize
        basis = np.array(basis, dtype=float)
        B_star, mu = self.GSO(basis)
        k = 1 # because python is 0 indexed
        n = len(basis)

        # Main loop
        while k < n:
            # Step 1: size reduction
            for j in reversed(range(k)):
                q = round(mu[k][j]) # more efficient than np.rint()

                if q != 0:
                    basis[k] = basis[k] - q * basis[j]
                    # update the values for mu_k,i for i = 1, ..., j
                    for i in range(j + 1):
                        mu[k, i] = mu[k, i] - q * mu[j, i]

            # Step 2: Lovasz Condition check
            norm_bk_star_sq = np.dot(B_star[k], B_star[k])
            norm_bk1_star_sq = np.dot(B_star[k-1], B_star[k-1])
        
            lovasz_bound = (delta - mu[k, k-1]**2) * norm_bk1_star_sq

            if norm_bk_star_sq < lovasz_bound:
                basis[k], basis[k-1] = basis[k-1].copy(), basis[k].copy()                    
                
                # update the GSO vectors
                B_star, mu = self.GSO(basis)

                k = max(1, k - 1)
            else:
                    k = k + 1


        return basis # should be reduced now
    
    def GSO(self, basis):
        """
        Compute the Gram-Schmidt Orthogonalization (GSO) of a basis.
        
        Returns:
            B_star: The orthogonalized basis vectors
            mu: The Gram-Schmidt coefficients matrix
        """
        n = len(basis)
        B_star = np.zeros_like(basis, dtype=float)
        mu = np.zeros((n, n), dtype=float)
        
        for i in range(n):
            B_star[i] = basis[i].astype(float)
            for j in range(i):
                # Compute mu_i,j = <b_i, b*_j> / <b*_j, b*_j>
                mu[i, j] = np.dot(basis[i], B_star[j]) / np.dot(B_star[j], B_star[j])
                # b*_i = b_i - sum(mu_i,j * b*_j)
                B_star[i] -= mu[i, j] * B_star[j]
            mu[i, i] = 1.0
        
        return B_star, mu


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("GGH/HNF Cryptosystem Implementation")
    print("=" * 60)
    n = 10
    
    ggh = GGHCryptosystem(n=n)
    
    print("\n### KEY GENERATION ###")
    ggh.generate_keys()
    
    print("\n### ENCRYPTION AND DECRYPTION ###")
    
    # Example message (integer coefficients)
    # message = np.array([3, -2, 1, 4, -1])
    message = np.random.randint(-4, 5, size=n)
    
    # Encrypt
    ciphertext = ggh.encrypt(message)
    
    # Decrypt
    decrypted = ggh.decrypt(ciphertext)
    
    # Verify
    print("\n### VERIFICATION ###")
    print(f"Original message:  {message}")
    print(f"Decrypted message: {decrypted}")
    print(f"Match: {np.array_equal(message, decrypted)}")
    
    # Show why the public key is "bad"
    print("\n### BASIS QUALITY COMPARISON ###")
    print(f"Private key Hadamard ratio: {ggh.hadamard_ratio(ggh.private_key):.6f}")
    print(f"Public key Hadamard ratio:  {ggh.hadamard_ratio(ggh.public_key):.6f}")
    print("\n(Closer to 1 is better; private key should be much better)")


    # print("\n### Hadamard ratio of public key (LLL) ###")
    # print(ggh.hadamard_ratio(ggh.LLL(ggh.public_key)))